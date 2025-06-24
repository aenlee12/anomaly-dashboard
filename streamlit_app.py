import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from sklearn.ensemble import IsolationForest
import plotly.express as px

st.set_page_config(page_title="Аномалии: списания & закрытие", layout="wide")

@st.cache_data
def load_and_prepare(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df['Списания %'] = pd.to_numeric(
        df['ЗЦ2_срок_качество_%'].str.replace(',', '.').str.rstrip('%'),
        errors='coerce'
    )
    df['Закрытие потребности %'] = pd.to_numeric(
        df['Закрытие потребности_%'].str.replace(',', '.').str.rstrip('%'),
        errors='coerce'
    )
    df['Продажа с ЗЦ сумма'] = pd.to_numeric(
        df['Продажа_с_ЗЦ_сумма'], errors='coerce'
    ).fillna(0)
    df = df.dropna(subset=['Списания %','Закрытие потребности %','Группа','Name_tov'])

    def agg_group(g):
        total_sales = g['Продажа с ЗЦ сумма'].sum()
        waste = (np.average(g['Списания %'], weights=g['Продажа с ЗЦ сумма'])
                 if total_sales > 0 else g['Списания %'].mean())
        fill  = (np.average(g['Закрытие потребности %'], weights=g['Продажа с ЗЦ сумма'])
                 if total_sales > 0 else g['Закрытие потребности %'].mean())
        return pd.Series({
            'Списания %': waste,
            'Закрытие потребности %': fill,
            'Продажа с ЗЦ сумма': total_sales
        })

    df = df.groupby(['Категория','Группа','Name_tov'], as_index=False).apply(agg_group)
    df['group_mean_waste'] = df.groupby('Группа')['Списания %'].transform('mean')
    weighted = df.groupby('Группа').apply(
        lambda g: np.average(g['Списания %'], weights=g['Продажа с ЗЦ сумма'])
        if g['Продажа с ЗЦ сумма'].sum()>0 else g['Списания %'].mean()
    ).to_dict()
    df['group_weighted_mean_waste'] = df['Группа'].map(weighted)
    return df

@st.cache_data
def score_anomalies(df):
    df = df.copy()
    df['anomaly_score'] = 0.0
    for grp, sub in df.groupby('Группа'):
        X = sub[['Списания %','Закрытие потребности %']]
        iso = IsolationForest(contamination=0.05, random_state=42,
                              n_jobs=-1, n_estimators=50)
        iso.fit(X)
        raw = iso.decision_function(X)
        df.loc[sub.index, 'anomaly_score'] = -raw
    df['anomaly_severity'] = df['anomaly_score'].abs()
    df['sales_share_in_group'] = (
        df['Продажа с ЗЦ сумма'] /
        df.groupby('Группа')['Продажа с ЗЦ сумма'].transform('sum')
    ).fillna(0)
    df['combined_score'] = df['anomaly_severity'] * df['sales_share_in_group']
    return df

def display_anomaly_table(df, title):
    st.subheader(f"{title}  (найдено {len(df)})")
    cols = [
        'Категория','Группа','Name_tov',
        'Списания %','group_mean_waste','group_weighted_mean_waste',
        'Закрытие потребности %','Продажа с ЗЦ сумма',
        'anomaly_severity','combined_score'
    ]
    rename_map = {
        'group_mean_waste':            'Среднее в группе %',
        'group_weighted_mean_waste':   'Средневзв. в группе %',
        'anomaly_severity':            'Степень аномалии',
        'combined_score':              'Скор в группе'
    }
    styler = (
        df[cols]
        .rename(columns=rename_map)
        .style.format({
            'Списания %':                     '{:.1f}',
            'Среднее в группе %':             '{:.1f}',
            'Средневзв. в группе %':          '{:.1f}',
            'Закрытие потребности %':         '{:.1f}',
            'Продажа с ЗЦ сумма':             '{:.0f}',
            'Степень аномалии':               '{:.3f}',
            'Скор в группе':                  '{:.3f}'
        })
        .background_gradient(subset=['Списания %'],            cmap='Reds')
        .background_gradient(subset=['Закрытие потребности %'],cmap='Blues')
        .background_gradient(subset=['Скор в группе'],        cmap='Purples')
    )
    st.dataframe(styler, use_container_width=True)

def main():
    st.title("Анализ аномалий: списания и закрытие потребности")

    uploaded = st.file_uploader("Загрузите CSV-файл с периодами", type="csv")
    if not uploaded:
        return

    df = load_and_prepare(uploaded)
    df = score_anomalies(df)

    # Фильтрация по категориям
    st.sidebar.header("Фильтрация по категориям")
    cats = sorted(df['Категория'].unique())
    sel_cats = st.sidebar.multiselect("Категории", cats, default=cats)
    df_cat = df[df['Категория'].isin(sel_cats)]

    # Поиск и фильтрация по группам
    st.sidebar.header("Фильтрация по группам")
    grps = sorted(df_cat['Группа'].unique())
    grp_query = st.sidebar.text_input("Поиск групп", "")
    grps_filtered = [g for g in grps if grp_query.lower() in g.lower()]
    sel_grps = st.sidebar.multiselect("Группы", grps_filtered, default=grps_filtered)
    df = df_cat[df_cat['Группа'].isin(sel_grps)]

    # Пресеты чувствительности
    sale_min, sale_max = df['Продажа с ЗЦ сумма'].min(), df['Продажа с ЗЦ сумма'].max()
    preset = st.sidebar.radio("Пресет чувствительности", [
        "Нет (ручная настройка)",
        "Слабая чувствительность",
        "Средняя чувствительность",
        "Высокая чувствительность"
    ])
    if preset == "Слабая чувствительность":
        sale_def, lw_def, lf_def, hw_def, hf_def = (
            (sale_min, sale_max), (0.5, 15.0), (5.0, 85.0), 15.0, 60.0
        )
    elif preset == "Средняя чувствительность":
        sale_def, lw_def, lf_def, hw_def, hf_def = (
            (sale_min, sale_max), (0.5, 8.0), (10.0, 75.0), 20.0, 80.0
        )
    else:  # Высокая или Нет
        sale_def, lw_def, lf_def, hw_def, hf_def = (
            (sale_min, sale_max), (0.5, 5.0), (20.0, 60.0), 25.0, 90.0
        )

    # Фильтры по выручке
    st.sidebar.header("Фильтры по выручке")
    sale_range = st.sidebar.slider(
        "Выручка (₽)",
        sale_min, sale_max,
        sale_def,
        step=1
    )
    sale_min_in = st.sidebar.number_input(
        "Мин. выручка (₽)",
        sale_min, sale_max,
        value=sale_range[0]
    )
    sale_max_in = st.sidebar.number_input(
        "Макс. выручка (₽)",
        sale_min, sale_max,
        value=sale_range[1]
    )

    # Низкие списания + низкое закрытие
    st.sidebar.header("Низкие списания + низкое закрытие")
    lw_slider = st.sidebar.slider(
        "Списания % (диапазон)",
        0.0, 100.0,
        lw_def,
        step=0.1
    )
    lw_min = st.sidebar.number_input(
        "Мин. списания %",
        0.0, 100.0,
        value=lw_slider[0]
    )
    lw_max = st.sidebar.number_input(
        "Макс. списания %",
        0.0, 100.0,
        value=lw_slider[1]
    )
    lf_slider = st.sidebar.slider(
        "Закрытие % (диапазон)",
        0.0, 100.0,
        lf_def,
        step=0.1
    )
    lf_min = st.sidebar.number_input(
        "Мин. закрытие %",
        0.0, 100.0,
        value=lf_slider[0]
    )
    lf_max = st.sidebar.number_input(
        "Макс. закрытие %",
        0.0, 100.0,
        value=lf_slider[1]
    )

    # Высокие списания + высокое закрытие
    st.sidebar.header("Высокие списания + высокое закрытие")
    hw_slider = st.sidebar.slider(
        "Порог списания %",
        0.0, 200.0,
        hw_def,
        step=0.1
    )
    hw_thr = st.sidebar.number_input(
        "Порог списания % вручную",
        0.0, 200.0,
        value=hw_slider
    )
    hf_slider = st.sidebar.slider(
        "Порог закрытия %",
        0.0, 200.0,
        hf_def,
        step=0.1
    )
    hf_thr = st.sidebar.number_input(
        "Порог закрытия % вручную",
        0.0, 200.0,
        value=hf_slider
    )

    # Применяем фильтры
    df = df[
        (df['Продажа с ЗЦ сумма'] >= sale_min_in) &
        (df['Продажа с ЗЦ сумма'] <= sale_max_in)
    ]
    low_df = df[
        df['Списания %'].between(lw_min, lw_max) &
        df['Закрытие потребности %'].between(lf_min, lf_max)
    ].sort_values('combined_score', ascending=False)
    high_df = df[
        (df['Списания %'] >= hw_thr) &
        (df['Закрытие потребности %'] >= hf_thr)
    ].sort_values('combined_score', ascending=False)

    display_anomaly_table(low_df,  "Низкие списания + низкое закрытие")
    display_anomaly_table(high_df, "Высокие списания + высокое закрытие")

    mask = df.index.isin(pd.concat([low_df, high_df]).index)
    df_plot = df.copy()
    df_plot['Статус'] = np.where(mask, 'Аномалия', 'Норма')
    fig = px.scatter(
        df_plot,
        x='Списания %',
        y='Закрытие потребности %',
        color='Статус',
        size='Продажа с ЗЦ сумма',
        opacity=0.6,
        hover_data=['Name_tov','Группа'],
        color_discrete_map={'Норма':'lightgrey','Аномалия':'crimson'}
    )
    st.plotly_chart(fig, use_container_width=True)

    buf = BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        low_df.to_excel(writer, sheet_name='Низкие списания + низкое закрытие', index=False)
        high_df.to_excel(writer, sheet_name='Высокие списания + высокое закрытие', index=False)
    buf.seek(0)
    st.download_button(
        "Скачать результаты в Excel",
        buf,
        "anomalies_group.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

if __name__ == "__main__":
    main()
