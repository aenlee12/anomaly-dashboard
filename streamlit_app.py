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

    # Переименование ключевых колонок
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
    for req in ['Категория','Группа','Name_tov','Формат','Склад']:
        if req not in df.columns:
            raise KeyError(f"Missing column '{req}'")

    # Продажа с ЗЦ сумма
    sale_col = next((c for c in df.columns if 'продажа' in c.lower() and 'сумма' in c.lower()), None)
    df['Продажа с ЗЦ сумма'] = pd.to_numeric(df[sale_col], errors='coerce').fillna(0)

    # Закрытие потребности %
    fill_col = next((c for c in df.columns if 'закрытие' in c.lower()), None)
    df['Закрытие потребности %'] = (
        pd.to_numeric(
            df[fill_col].astype(str).str.replace(',', '.').str.rstrip('%'),
            errors='coerce'
        ) * 100
    )

    # Списания %
    zc2  = next((c for c in df.columns if c.lower().startswith('зц2')), None)
    srok = next((c for c in df.columns if c.lower() == 'срок'), None)
    kach = next((c for c in df.columns if c.lower() == 'качество'), None)
    for c in (zc2, srok, kach):
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    df['Списания %'] = np.where(
        df['Продажа с ЗЦ сумма']>0,
        (df[zc2]+df[srok]+df[kach])/df['Продажа с ЗЦ сумма']*100,
        0
    )

    df = df.dropna(subset=['Категория','Группа','Name_tov','Формат','Склад','Списания %','Закрытие потребности %'])
    return df


def score_anomalies(df):
    df = df.copy()
    df['anomaly_score'] = 0.0
    for grp, sub in df.groupby('Группа'):
        if len(sub)<2: continue
        X = sub[['Списания %','Закрытие потребности %']]
        iso = IsolationForest(contamination=0.05, random_state=42, n_estimators=30, n_jobs=-1)
        iso.fit(X)
        df.loc[sub.index,'anomaly_score'] = -iso.decision_function(X)
    df['severity'] = df['anomaly_score'].abs()
    df['share']    = df['Продажа с ЗЦ сумма'] / df.groupby('Группа')['Продажа с ЗЦ сумма'].transform('sum').fillna(1)
    df['combined'] = df['severity']*df['share']
    return df


@st.cache_data(show_spinner=False)
def process_data(uploaded):
    return score_anomalies(load_and_prepare(uploaded))


def display_anomaly_table(df: pd.DataFrame, title: str):
    """
    Таблица аномалий с сортировкой по Скору (combined) и уникальными
    средними комбо-метриками по ['Склад','Формат','Категория','Группа'].
    """
    df = df.copy()
    combo_keys = ['Склад','Формат','Категория','Группа']

    # Простой и взвешенный mean по combo_keys
    df['Среднее комбо %'] = df.groupby(combo_keys)['Списания %'].transform('mean')
    def wavg(x):
        w = x['Продажа с ЗЦ сумма']; v = x['Списания %']
        return np.average(v, weights=w) if w.sum()>0 else v.mean()
    w = df.groupby(combo_keys).apply(wavg).rename('Ср.взв. комбо %')
    df = df.join(w, on=combo_keys)

    # Сортируем по Скору (combined) убывающе
    df = df.sort_values('combined', ascending=False)

    st.subheader(f"{title} — {len(df)}")
    display_cols = [
        'Категория','Группа','Формат','Склад','Name_tov',
        'Списания %','Среднее комбо %','Ср.взв. комбо %',
        'Закрытие потребности %','Продажа с ЗЦ сумма',
        'severity','combined'
    ]
    styled = (
        df[display_cols]
        .rename(columns={'severity':'Степень аномалии','combined':'Скор'})
        .style.format({
            'Списания %':'{:.1f}','Среднее комбо %':'{:.1f}','Ср.взв. комбо %':'{:.1f}',
            'Закрытие потребности %':'{:.1f}','Продажа с ЗЦ сумма':'{:.0f}',
            'Степень аномалии':'{:.3f}','Скор':'{:.3f}'
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

    full_df = process_data(uploaded)

    # Sidebar: фильтрация по складам, форматам, категориям, группам
    df = full_df.copy()
    sb = st.sidebar
    sb.header("Фильтрация")

    sel_whs  = sb.multiselect("Склады",  sorted(df['Склад'].unique()),              default=sorted(df['Склад'].unique()))
    sel_fmts = sb.multiselect("Форматы", sorted(df[df['Склад'].isin(sel_whs)]['Формат'].unique()), default=None)
    df = df[df['Склад'].isin(sel_whs) & df['Формат'].isin(sel_fmts or df['Формат'].unique())]

    sel_cats = sb.multiselect("Категории", sorted(df['Категория'].unique()),          default=sorted(df['Категория'].unique()))
    sel_grps = sb.multiselect("Группы",     sorted(df[df['Категория'].isin(sel_cats)]['Группа'].unique()), default=None)
    df = df[df['Категория'].isin(sel_cats) & df['Группа'].isin(sel_grps or df['Группа'].unique())]

    if df.empty:
        st.warning("Нет данных после фильтров")
        return

    # Фильтр по выручке
    smin, smax = int(df['Продажа с ЗЦ сумма'].min()), int(df['Продажа с ЗЦ сумма'].max())
    rev_min, rev_max = sb.slider("Выручка (₽)", smin, smax, (smin, smax), step=1)
    df = df[df['Продажа с ЗЦ сумма'].between(rev_min, rev_max)]

    # Чувствительность
    sb.markdown("---")
    sb.header("Чувствительность")
    presets = {
        "Нет":     ((0.0,100.0),(0.0,100.0),  0.0,  0.0),
        "Слабая":  ((0.5,15.0),(5.0,85.0),   15.0, 60.0),
        "Средняя": ((0.5,8.0),(10.0,75.0),   20.0, 80.0),
        "Высокая": ((0.5,5.0),(20.0,60.0),   25.0, 90.0),
    }
    p = sb.radio("Пресет", list(presets.keys()), index=2)
    (lw_min,lw_max),(cl_min,cl_max),hw_thr,hf_thr = presets[p]

    sb.subheader("Низкие списания + низкое закрытие")
    low_min, low_max     = sb.slider("Списания %", 0.0,100.0,(lw_min,lw_max),step=0.1)
    close_min, close_max = sb.slider("Закрытие %",   0.0,100.0,(cl_min,cl_max),step=0.1)

    sb.markdown("---")
    sb.subheader("Высокие списания + высокое закрытие")
    hw_thr = sb.number_input("Порог списания %",  0.0,200.0,hw_thr,step=0.1)
    hf_thr = sb.number_input("Порог закрытия %",  0.0,200.0,hf_thr,step=0.1)

    # Отбираем низкие и высокие аномалии из отфильтрованного df
    low_df  = df[df['Списания %'].between(low_min, low_max)     & df['Закрытие потребности %'].between(close_min, close_max)]
    high_df = df[(df['Списания %'] >= hw_thr) & (df['Закрытие потребности %'] >= hf_thr)]

    # 1) Таблицы аномалий (сортировка по Скору внутри display_anomaly_table)
    display_anomaly_table(low_df,  "Низкие списания + низкое закрытие (топ-100)")
    display_anomaly_table(high_df, "Высокие списания + высокое закрытие (топ-100)")

    # 2) Диаграмма всех SKU (не зависит от фильтра: используется full_df)
    st.subheader("Диаграмма всех SKU")
    bubble = full_df.copy()

    # Маски на полном пуле
    mask_anom = (
        (bubble['Списания %'].between(low_min, low_max) &
         bubble['Закрытие потребности %'].between(close_min, close_max))
        |
        (bubble['Списания %'] >= hw_thr) &
        (bubble['Закрытие потребности %'] >= hf_thr)
    )
    mask_report = bubble.index.isin(pd.concat([low_df, high_df]).index)

    bubble['Статус'] = np.where(mask_report, 'В отчете',
                          np.where(mask_anom,    'Аномалия','Норма'))

    fig = px.scatter(
        bubble,
        x='Закрытие потребности %',
        y='Списания %',
        color='Статус',
        size='Продажа с ЗЦ сумма',
        opacity=0.6,
        hover_data=['Name_tov','Группа','Формат','Склад'],
        color_discrete_map={'Норма':'lightgray','Аномалия':'crimson','В отчете':'purple'}
    )
    fig.update_xaxes(range=[0,100])
    fig.update_yaxes(range=[0,100])
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
