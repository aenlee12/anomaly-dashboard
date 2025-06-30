# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from sklearn.ensemble import IsolationForest
import plotly.express as px

st.set_page_config(page_title="Аномалии: списания & закрытие", layout="wide")


def load_and_prepare(uploaded):
    name = uploaded.name.lower()
    if name.endswith((".xls", ".xlsx")):
        df = pd.read_excel(uploaded, engine="openpyxl")
    else:
        df = pd.read_csv(uploaded)
    df.columns = df.columns.str.strip()

    # Переименование колонок
    col_map = {}
    if 'parent_group_name' in df.columns:
        col_map['parent_group_name'] = 'Категория'
    if 'group_name' in df.columns:
        col_map['group_name'] = 'Группа'
    fmt = next((c for c in df.columns if 'формат' in c.lower() or 'format' in c.lower()), None)
    if fmt:
        col_map[fmt] = 'Формат'
    skl = next((c for c in df.columns if 'name_sklad' in c.lower()), None) or \
          next((c for c in df.columns if 'sklad' in c.lower()), None)
    if skl:
        col_map[skl] = 'Склад'
    df = df.rename(columns=col_map)

    # Проверка обязательных колонок
    for req in ['Категория', 'Группа', 'Name_tov', 'Формат', 'Склад']:
        if req not in df.columns:
            raise KeyError(f"Missing column '{req}'")

    # Продажа с ЗЦ сумма
    sale_col = next((c for c in df.columns if 'продажа' in c.lower() and 'сумма' in c.lower()), None)
    if not sale_col:
        raise KeyError("Missing sales sum column")
    df['Продажа с ЗЦ сумма'] = pd.to_numeric(df[sale_col], errors='coerce').fillna(0)

    # Закрытие потребности %
    fill_col = next((c for c in df.columns if 'закрытие' in c.lower()), None)
    if not fill_col:
        raise KeyError("Missing fill percentage column")
    df['Закрытие потребности %'] = (
        pd.to_numeric(df[fill_col].astype(str).str.replace(',', '.').str.rstrip('%'),
                      errors='coerce') * 100
    )

    # Списания %
    zc2 = next((c for c in df.columns if c.lower().startswith('зц2')), None)
    srok = next((c for c in df.columns if c.lower() == 'срок'), None)
    kach = next((c for c in df.columns if c.lower() == 'качество'), None)
    if not all([zc2, srok, kach]):
        raise KeyError("Missing waste columns ZC2, срок, or качество")
    for c in (zc2, srok, kach):
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    df['Списания %'] = np.where(
        df['Продажа с ЗЦ сумма'] > 0,
        (df[zc2] + df[srok] + df[kach]) / df['Продажа с ЗЦ сумма'] * 100,
        0
    )

    df = df.dropna(subset=['Категория','Группа','Name_tov','Формат','Склад','Списания %','Закрытие потребности %'])

    # Агрегация по SKU → Формат → Склад
    grp_cols = ['Категория','Группа','Name_tov','Формат','Склад']
    grp = df.groupby(grp_cols)
    tot = grp['Продажа с ЗЦ сумма'].sum()
    waste = grp.apply(lambda x: np.average(x['Списания %'], weights=x['Продажа с ЗЦ сумма'])
                      if x['Продажа с ЗЦ сумма'].sum()>0 else x['Списания %'].mean())
    fillp = grp.apply(lambda x: np.average(x['Закрытие потребности %'], weights=x['Продажа с ЗЦ сумма'])
                      if x['Продажа с ЗЦ сумма'].sum()>0 else x['Закрытие потребности %'].mean())
    agg = pd.concat([tot,waste,fillp], axis=1).reset_index()
    agg.columns = grp_cols + ['Продажа с ЗЦ сумма','Списания %','Закрытие потребности %']

    # Групповые метрики
    agg['avg_waste_in_group'] = agg.groupby('Группа')['Списания %'].transform('mean')
    wmap = agg.groupby('Группа').apply(
        lambda x: np.average(x['Списания %'], weights=x['Продажа с ЗЦ сумма'])
                  if x['Продажа с ЗЦ сумма'].sum()>0 else x['Списания %'].mean()
    ).to_dict()
    agg['wavg_waste_in_group'] = agg['Группа'].map(wmap)

    return agg


def score_anomalies(df):
    df = df.copy()
    df['anomaly_score'] = 0.0
    for grp, sub in df.groupby('Группа'):
        if len(sub)<2:
            continue
        X = sub[['Списания %','Закрытие потребности %']]
        iso = IsolationForest(contamination=0.05, random_state=42, n_estimators=30, n_jobs=-1)
        iso.fit(X)
        df.loc[sub.index,'anomaly_score'] = -iso.decision_function(X)
    df['severity'] = df['anomaly_score'].abs()
    df['share'] = df['Продажа с ЗЦ сумма']/df.groupby('Группа')['Продажа с ЗЦ сумма'].transform('sum').fillna(1)
    df['combined'] = df['severity']*df['share']
    return df


@st.cache_data(show_spinner=False)
def process_data(uploaded):
    return score_anomalies(load_and_prepare(uploaded))


@st.cache_data(show_spinner=False)
def get_hierarchy_df(full_df):
    return (
        full_df
        .groupby(['Склад','Формат','Категория','Группа'])
        .agg({'Списания %':'mean','Закрытие потребности %':'mean','Продажа с ЗЦ сумма':'sum'})
        .reset_index()
    )


def display_anomaly_table(df, title):
    st.subheader(f"{title} — {len(df)}")
    rename = {
        'avg_waste_in_group':'Среднее в группе %',
        'wavg_waste_in_group':'Ср.взв. в группе %',
        'severity':'Степень аномалии',
        'combined':'Скор'
    }
    styled = (
        df.rename(columns=rename)[
            ['Категория','Группа','Формат','Склад','Name_tov',
             'Списания %','Среднее в группе %','Ср.взв. в группе %',
             'Закрытие потребности %','Продажа с ЗЦ сумма',
             'Степень аномалии','Скор']
        ]
        .style.format({
            'Списания %':'{:.1f}','Среднее в группе %':'{:.1f}',
            'Ср.взв. в группе %':'{:.1f}','Закрытие потребности %':'{:.1f}',
            'Продажа с ЗЦ сумма':'{:.0f}','Степень аномалии':'{:.3f}','Скор':'{:.3f}'
        })
        .background_gradient(subset=['Списания %'], cmap='Reds')
        .background_gradient(subset=['Закрытие потребности %'], cmap='Blues')
        .background_gradient(subset=['Скор'], cmap='Purples')
    )
    st.dataframe(styled, use_container_width=True)


def main():
    st.title("Аномалии: списания и закрытие потребности")

    uploaded = st.file_uploader("Загрузите CSV или Excel", type=['csv','xls','xlsx'])
    if not uploaded:
        return

    # 1) Загрузка и скоринг
    full_df = process_data(uploaded)

    # 2) Отладочный вывод
    st.write("**Первые 10 строк (отладка)**")
    st.dataframe(full_df[['Name_tov','Формат','Склад','Списания %','Закрытие потребности %']].head(10),
                 use_container_width=True)

    # 3) Sidebar — фильтрация
    df = full_df.copy()
    sb = st.sidebar
    sb.header("Фильтрация")

    # 3.1 Категории → группы
    cats = sorted(df['Категория'].unique())
    sel_cats = sb.multiselect("Категории", cats, default=cats)
    grps = sorted(df[df['Категория'].isin(sel_cats)]['Группа'].unique())
    sel_grps = sb.multiselect("Группы", grps, default=grps)
    df = df[df['Категория'].isin(sel_cats)&df['Группа'].isin(sel_grps)]

    # 3.2 Выручка
    smin, smax = int(df['Продажа с ЗЦ сумма'].min()), int(df['Продажа с ЗЦ сумма'].max())
    sel_rng = sb.slider("Выручка (₽)", smin, smax, (smin, smax), step=1)
    min_rev = sb.number_input("Мин. выручка (₽)", smin, smax, sel_rng[0], step=1)
    max_rev = sb.number_input("Макс. выручка (₽)", smin, smax, sel_rng[1], step=1)
    df = df[df['Продажа с ЗЦ сумма'].between(min_rev, max_rev)]

    # 3.3 Чувствительность
    sb.markdown("---")
    sb.header("Чувствительность")
    presets = {
        "Нет":     ((0.0,100.0),(0.0,100.0),0.0,0.0),
        "Слабая":  ((0.5,15.0),(5.0,85.0),15.0,60.0),
        "Средняя": ((0.5,8.0),(10.0,75.0),20.0,80.0),
        "Высокая": ((0.5,5.0),(20.0,60.0),25.0,90.0)
    }
    preset = sb.radio("Пресет", list(presets.keys()), index=2)
    lw_def, lf_def, hw_def, hf_def = presets[preset]

    sb.subheader("Низкие списания + низкое закрытие")
    low_min, low_max = sb.slider("Списания % (диапазон)", 0.0, 100.0, lw_def, step=0.1)
    close_min, close_max = sb.slider("Закрытие % (диапазон)", 0.0, 100.0, lf_def, step=0.1)

    sb.markdown("---")
    sb.subheader("Высокие списания + высокое закрытие")
    hw_thr = sb.number_input("Порог списания %", 0.0, 200.0, hw_def, step=0.1)
    hf_thr = sb.number_input("Порог закрытия %", 0.0, 200.0, hf_def, step=0.1)

    low_df = df[df['Списания %'].between(low_min, low_max) &
                df['Закрытие потребности %'].between(close_min, close_max)]
    high_df = df[(df['Списания %']>=hw_thr)&(df['Закрытие потребности %']>=hf_thr)]

    # 4) Таблицы аномалий
    display_anomaly_table(low_df.sort_values('combined', ascending=False).head(100),
                          "Низкие списания + низкое закрытие (топ-100)")
    display_anomaly_table(high_df.sort_values('combined', ascending=False).head(100),
                          "Высокие списания + высокое закрытие (топ-100)")

    # 5) Диаграмма всех SKU (0–150%)
    st.subheader("Диаграмма всех SKU")
    mask_anom   = df.index.isin(pd.concat([low_df,high_df]).index)
    mask_report = df.index.isin(pd.concat([
        low_df.sort_values('combined', ascending=False).head(100),
        high_df.sort_values('combined', ascending=False).head(100)
    ]).index)
    df_plot = df.copy()
    df_plot['Статус'] = np.where(mask_report,'В отчете',
                          np.where(mask_anom,'Аномалия','Норма'))

    fig = px.scatter(df_plot,
                     x='Закрытие потребности %',
                     y='Списания %',
                     color='Статус',
                     size='Продажа с ЗЦ сумма',
                     opacity=0.6,
                     hover_data=['Name_tov','Группа','Формат','Склад'],
                     color_discrete_map={'Норма':'lightgray','Аномалия':'crimson','В отчете':'purple'})
    fig.update_xaxes(range=[0,150])
    fig.update_yaxes(range=[0,150])
    st.plotly_chart(fig, use_container_width=True)

    # 6) Иерархическая фильтрация (каскадно по comp)
    st.subheader("Иерархическая фильтрация")
    comp = get_hierarchy_df(full_df)

    hdf = comp.copy()
    wh_opts = sorted(comp['Склад'].unique())
    sel_whs = st.multiselect("Склады", wh_opts, default=wh_opts)
    if sel_whs:
        hdf = hdf[hdf['Склад'].isin(sel_whs)]

    fmt_opts = sorted(hdf['Формат'].unique())
    sel_fmts = st.multiselect("Форматы", fmt_opts, default=fmt_opts)
    if sel_fmts:
        hdf = hdf[hdf['Формат'].isin(sel_fmts)]

    cat_opts = sorted(hdf['Категория'].unique())
    sel_cats2 = st.multiselect("Категории", cat_opts, default=cat_opts)
    if sel_cats2:
        hdf = hdf[hdf['Категория'].isin(sel_cats2)]

    grp_opts = sorted(hdf['Группа'].unique())
    sel_grps2 = st.multiselect("Группы", grp_opts, default=grp_opts)
    if sel_grps2:
        hdf = hdf[hdf['Группа'].isin(sel_grps2)]

    st.dataframe(hdf[['Склад','Формат','Категория','Группа',
                      'Списания %','Закрытие потребности %','Продажа с ЗЦ сумма']],
                 use_container_width=True)

    # 7) Экспорт
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as w:
        low_df.to_excel(w, sheet_name='Низкие', index=False)
        high_df.to_excel(w, sheet_name='Высокие', index=False)
    buf.seek(0)
    st.download_button("Скачать Excel", buf,
                       "anomalies.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


if __name__ == "__main__":
    main()
