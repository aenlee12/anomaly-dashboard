import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from sklearn.ensemble import IsolationForest
import plotly.express as px

st.set_page_config(page_title="Аномалии: списания & закрытие", layout="wide")

# Убираем кэш, чтобы всегда обновлять данные при каждом деплое

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
    skl = next((c for c in df.columns if 'name_sklad' in c.lower()), None)
    if skl:
        col_map[skl] = 'Склад'
    else:
        skl = next((c for c in df.columns if 'sklad' in c.lower()), None)
        if skl:
            col_map[skl] = 'Склад'
    df = df.rename(columns=col_map)

    # Проверка обязательных колонок
    for req in ['Категория', 'Группа', 'Name_tov', 'Формат', 'Склад']:
        if req not in df.columns:
            raise KeyError(f"Missing column '{req}' in input file")

    # Выручка
    sale_col = next((c for c in df.columns if 'продажа' in c.lower() and 'сумма' in c.lower()), None)
    if not sale_col:
        raise KeyError("Missing sales sum column")
    df['Продажа с ЗЦ сумма'] = pd.to_numeric(df[sale_col], errors='coerce').fillna(0)

    # Закрытие потребности
    fill_col = next((c for c in df.columns if 'закрытие' in c.lower()), None)
    if not fill_col:
        raise KeyError("Missing fill percentage column")
    df['Закрытие потребности %'] = (
        pd.to_numeric(
            df[fill_col].astype(str).str.replace(',', '.').str.rstrip('%'),
            errors='coerce'
        ) * 100
    )

    # Списания
    col_zc2 = next((c for c in df.columns if c.lower().startswith('зц2')), None)
    col_srok = next((c for c in df.columns if c.lower() == 'срок'), None)
    col_kach = next((c for c in df.columns if c.lower() == 'качество'), None)
    if not (col_zc2 and col_srok and col_kach):
        raise KeyError("Missing waste columns ZC2, срок, or качество")
    for c in (col_zc2, col_srok, col_kach):
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    df['Списания %'] = np.where(
        df['Продажа с ЗЦ сумма'] > 0,
        (df[col_zc2] + df[col_srok] + df[col_kach]) / df['Продажа с ЗЦ сумма'] * 100,
        0
    )

    df = df.dropna(subset=['Категория', 'Группа', 'Name_tov', 'Формат', 'Склад', 'Списания %', 'Закрытие потребности %'])

    # Агрегация по SKU, Формат и Склад
    grp_cols = ['Категория', 'Группа', 'Name_tov', 'Формат', 'Склад']
    grp = df.groupby(grp_cols)
    tot = grp['Продажа с ЗЦ сумма'].sum()
    waste = grp.apply(lambda x: np.average(x['Списания %'], weights=x['Продажа с ЗЦ сумма'])
                      if x['Продажа с ЗЦ сумма'].sum() > 0 else x['Списания %'].mean())
    fill = grp.apply(lambda x: np.average(x['Закрытие потребности %'], weights=x['Продажа с ЗЦ сумма'])
                     if x['Продажа с ЗЦ сумма'].sum() > 0 else x['Закрытие потребности %'].mean())
    agg = pd.concat([tot, waste, fill], axis=1).reset_index()
    agg.columns = grp_cols + ['Продажа с ЗЦ сумма', 'Списания %', 'Закрытие потребности %']

    # Групповые метрики
    agg['group_mean_waste'] = agg.groupby('Группа')['Списания %'].transform('mean')
    wmap = agg.groupby('Группа').apply(
        lambda x: np.average(x['Списания %'], weights=x['Продажа с ЗЦ сумма'])
        if x['Продажа с ЗЦ сумма'].sum() > 0 else x['Списания %'].mean()
    ).to_dict()
    agg['group_weighted_mean_waste'] = agg['Группа'].map(wmap)

    return agg

# Оценка аномалий без кэша

def score_anomalies(df):
    df = df.copy()
    df['anomaly_score'] = 0.0
    for grp, sub in df.groupby('Группа'):
        if len(sub) < 2:
            continue
        X = sub[['Списания %', 'Закрытие потребности %']]
        iso = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1, n_estimators=30)
        iso.fit(X)
        df.loc[sub.index, 'anomaly_score'] = -iso.decision_function(X)
    df['anomaly_severity'] = df['anomaly_score'].abs()
    df['sales_share_in_group'] = df['Продажа с ЗЦ сумма'] / df.groupby('Группа')['Продажа с ЗЦ сумма'].transform('sum').fillna(1)
    df['combined_score'] = df['anomaly_severity'] * df['sales_share_in_group']
    return df


def display_anomaly_table(df, title):
    st.subheader(f"{title} (найдено {len(df)})")
    cols = ['Категория', 'Группа', 'Формат', 'Склад', 'Name_tov', 'Списания %',
            'group_mean_waste', 'group_weighted_mean_waste', 'Закрытие потребности %',
            'Продажа с ЗЦ сумма', 'anomaly_severity', 'combined_score']
    rename_map = {
        'group_mean_waste': 'Среднее в группе %',
        'group_weighted_mean_waste': 'Средневзв. в группе %',
        'anomaly_severity': 'Степень аномалии',
        'combined_score': 'Скор в группе'
    }
    styler = (
        df[cols]
        .rename(columns=rename_map)
        .style
        .format({
            'Списания %': '{:.1f}',
            'Среднее в группе %': '{:.1f}',
            'Средневзв. в группе %': '{:.1f}',
            'Закрытие потребности %': '{:.1f}',
            'Продажа с ЗЦ сумма': '{:.0f}',
            'Степень аномалии': '{:.3f}',
            'Скор в группе': '{:.3f}'
        })
        .background_gradient(subset=['Списания %'], cmap='Reds')
        .background_gradient(subset=['Закрытие потребности %'], cmap='Blues')
        .background_gradient(subset=['Скор в группе'], cmap='Purples')
    )
    st.dataframe(styler, use_container_width=True)


def main():
    st.title("Аномалии: списания и закрытие потребности")
    uploaded = st.file_uploader("Загрузите CSV/Excel", type=['csv', 'xls', 'xlsx'])
    if not uploaded:
        return

    full_df = score_anomalies(load_and_prepare(uploaded))
    # Для отладки: покажем первые 10 строк с разными конечными метриками
    st.write("**Первые 10 комбинаций (для отладки)**", full_df[['Name_tov','Формат','Склад','Списания %','Закрытие потребности %']].head(10))
    df = full_df.copy()

    sb = st.sidebar
    sb.header("Фильтрация")

    cats = sorted(df['Категория'].unique())
    sel_cats = sb.multiselect("Категории", cats, default=cats)
    grps = sorted(df.loc[df['Категория'].isin(sel_cats), 'Группа'].unique())
    sel_grps = sb.multiselect("Группы", grps, default=grps)

    fmts = sorted(df['Формат'].unique())
    sel_fmts = sb.multiselect("Форматы ТТ", fmts, default=[])
    whs = sorted(df['Склад'].unique())
    sel_whs = sb.multiselect("Склады", whs, default=[])

    df = df[df['Категория'].isin(sel_cats) & df['Группа'].isin(sel_grps)]
    if sel_fmts:
        df = df[df['Формат'].isin(sel_fmts)]
    if sel_whs:
        df = df[df['Склад'].isin(sel_whs)]
    if df.empty:
        st.warning("Нет данных после фильтров")
        return

    smin, smax = int(df['Продажа с ЗЦ сумма'].min()), int(df['Продажа с ЗЦ сумма'].max())
    sel_rng = sb.slider("Выручка (₽)", smin, smax, (smin, smax), step=1)
    min_rev = sb.number_input("Мин. выручка (₽)", smin, smax, value=sel_rng[0], step=1)
    max_rev = sb.number_input("Макс. выручка (₽)", smin, smax, value=sel_rng[1], step=1)
    df = df[df['Продажа с ЗЦ сумма'].between(min_rev, max_rev)]

    sb.markdown("---")
    sb.header("Чувствительность")
    preset = sb.radio("Пресет", ["Нет", "Слабая", "Средняя", "Высокая"])
    defs = {
        "Нет":     ((0.0, 100.0), (0.0, 100.0), 0.0, 0.0),
        "Слабая":  ((0.5, 15.0),  (5.0, 85.0),  15.0, 60.0),
        "Средняя": ((0.5, 8.0),   (10.0, 75.0), 20.0, 80.0),
        "Высокая": ((0.5, 5.0),   (20.0, 60.0), 25.0, 90.0)
    }
    lw_def, lf_def, hw_def, hf_def = defs[preset]

    sb.subheader("Низкие списания + низкое закрытие")
    low_range = sb.slider("Списания % (диапазон)", 0.0, 100.0, lw_def, step=0.1)
    low_min = sb.number_input("Мин. списания % (число)", 0.0, 100.0, value=low_range[0], step=0.1)
    low_max = sb.number_input("Макс. списания % (число)", 0.0, 100.0, value=low_range[1], step=0.1)
    close_range = sb.slider("Закрытие % (диапазон)", 0.0, 100.0, lf_def, step=0.1)
    close_min = sb.number_input("Мин. закрытие % (число)", 0.0, 100.0, value=close_range[0], step=0.1)
    close_max = sb.number_input("Макс. закрытие % (число)", 0.0, 100.0, value=close_range[1], step=0.1)

    sb.markdown("---")
    sb.subheader("Высокие списания + высокое закрытие")
    hw_slider = sb.slider("Порог списания %", 0.0, 200.0, hw_def, step=0.1)
    hf_slider = sb.slider("Порог закрытия %", 0.0, 200.0, hf_def, step=0.1)
    hw_thr = sb.number_input("Порог списания % (число)", 0.0, 200.0, value=hw_slider, step=0.1)
    hf_thr = sb.number_input("Порог закрытия % (число)", 0.0, 200.0, value=hf_slider, step=0.1)

    low_df = df[df['Списания %'].between(low_min, low_max) & df['Закрытие потребности %'].between(close_min, close_max)]
    high_df = df[(df['Списания %'] >= hw_thr) & (df['Закрытие потребности %'] >= hf_thr)]

    display_anomaly_table(low_df.sort_values('combined_score', ascending=False), "Низкие списания + низкое закрытие")
    display_anomaly_table(high_df.sort_values('combined_score', ascending=False), "Высокие списания + высокое закрытие")

    # Scatter plot
    mask = df.index.isin(pd.concat([low_df, high_df]).index)
    df_plot = df.copy()
    df_plot['Статус'] = np.where(mask, 'Аномалия', 'Норма')
    fig = px.scatter(
        df_plot,
        x='Списания %', y='Закрытие потребности %',
        color='Статус', size='Продажа с ЗЦ сумма', opacity=0.6,
        hover_data=['Name_tov','Группа','Формат','Склад'],
        color_discrete_map={'Норма':'lightgray','Аномалия':'crimson'}
    )
    st.plotly_chart(fig, use_container_width=True)

    # Экспорт
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        low_df.to_excel(writer, sheet_name='Низкие', index=False)
        high_df.to_excel(writer, sheet_name='Высокие', index=False)
    buf.seek(0)
    st.download_button("Скачать Excel", buf, "anomalies.xlsx", 
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # Сравнение по всем форматам/складам
    st.header("Сравнение по форматам и складам (все данные)")
    with st.expander("Показать/скрыть полное сравнение"):
        comp = full_df.groupby(['Формат', 'Склад']).agg({
            'Списания %': 'mean',
            'Закрытие потребности %': 'mean',
            'Продажа с ЗЦ сумма': 'sum'
        }).reset_index().sort_values('Продажа с ЗЦ сумма', ascending=False)

        st.dataframe(
            comp.style.format({
                'Списания %': '{:.1f}',
                'Закрытие потребности %': '{:.1f}',
                'Продажа с ЗЦ сумма': '{:.0f}'
            })
            .background_gradient(subset=['Списания %'], cmap='Reds')
            .background_gradient(subset=['Закрытие потребности %'], cmap='Blues'),
            use_container_width=True
        )

if __name__ == "__main__":
    main()
