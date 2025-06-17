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
    return df.dropna(subset=['Списания %','Закрытие потребности %'])

@st.cache_data
def score_anomalies(df):
    X = df[['Списания %','Закрытие потребности %']]
    iso = IsolationForest(contamination=0.05, random_state=42)
    iso.fit(X)
    raw = iso.decision_function(X)
    df['anomaly_score']    = -raw
    df['anomaly_severity'] = df['anomaly_score'].abs()
    df['combined_score']   = df['anomaly_severity'] * df['Продажа с ЗЦ сумма']
    return df

def display_anomaly_table(df, title):
    st.subheader(f"{title}  (найдено {len(df)})")
    cols = [
        'Категория','Группа','Name_tov',
        'Списания %','Закрытие потребности %',
        'Продажа с ЗЦ сумма','anomaly_severity','combined_score'
    ]
    styler = df[cols].style.format({
        'Списания %': '{:.1f}',
        'Закрытие потребности %': '{:.1f}',
        'Продажа с ЗЦ сумма': '{:.0f}',
        'anomaly_severity': '{:.3f}',
        'combined_score': '{:.0f}',
    }).background_gradient(subset=['Списания %'], cmap='Reds') \
      .background_gradient(subset=['Закрытие потребности %'], cmap='Blues') \
      .background_gradient(subset=['combined_score'], cmap='Purples')
    st.dataframe(styler, use_container_width=True)

def main():
    st.title("Анализ аномалий: списания и закрытие потребности")

    uploaded = st.file_uploader("Загрузите CSV-файл", type="csv")
    if not uploaded:
        return

    # 1. Data prep + scoring
    df = load_and_prepare(uploaded)
    df = score_anomalies(df)

    # 2. Sidebar: пресеты и слайдеры
    sale_min, sale_max = float(df['Продажа с ЗЦ сумма'].min()), float(df['Продажа с ЗЦ сумма'].max())
    preset = st.sidebar.radio("Пресет чувствительности", [
        "Нет (ручная настройка)",
        "Слабая чувствительность",
        "Средняя чувствительность",
        "Высокая чувствительность"
    ])
    if preset == "Слабая чувствительность":
        sale_def, low_waste_def, low_fill_def, high_waste_def, high_fill_def = (
            (sale_min, sale_max), (0.5,15.0), (5.0,85.0), 15.0, 60.0
        )
    elif preset == "Средняя чувствительность":
        sale_def, low_waste_def, low_fill_def, high_waste_def, high_fill_def = (
            (sale_min, sale_max), (0.5,8.0), (10.0,75.0), 20.0, 80.0
        )
    elif preset == "Высокая чувствительность":
        sale_def, low_waste_def, low_fill_def, high_waste_def, high_fill_def = (
            (sale_min, sale_max), (0.5,5.0), (20.0,60.0), 40.0, 80.0
        )
    else:
        sale_def, low_waste_def, low_fill_def, high_waste_def, high_fill_def = (
            (sale_min, sale_max), (0.5,8.0), (10.0,75.0), 20.0, 80.0
        )

    st.sidebar.header("Настройки фильтрации")
    sale_range = st.sidebar.slider("Сумма продаж (руб.)", sale_min, sale_max, sale_def)
    st.sidebar.divider()
    low_waste = st.sidebar.slider("Низкие списания % диапазон", 0.0, 100.0, low_waste_def)
    low_fill  = st.sidebar.slider("Низкое закрытие % диапазон", 0.0, 100.0, low_fill_def)
    st.sidebar.divider()
    high_waste = st.sidebar.slider("Высокие списания % порог", 0.0, 200.0, high_waste_def)
    high_fill  = st.sidebar.slider("Высокое закрытие % порог", 0.0, 200.0, high_fill_def)

    # 3. Фильтрация + сортировка
    df = df[
        (df['Продажа с ЗЦ сумма'] >= sale_range[0]) &
        (df['Продажа с ЗЦ сумма'] <= sale_range[1])
    ]
    low_df  = df[
        df['Списания %'].between(*low_waste) &
        df['Закрытие потребности %'].between(*low_fill)
    ].sort_values('combined_score', ascending=False)
    high_df = df[
        (df['Списания %'] >= high_waste) &
        (df['Закрытие потребности %'] >= high_fill)
    ].sort_values('combined_score', ascending=False)

    # 4. Таблицы
    display_anomaly_table(low_df,  "Низкие списания + низкое закрытие потребности")
    display_anomaly_table(high_df, "Высокие списания + высокое закрытие потребности")

    # 5. Сравнительный график в самом низу
    # отметим аномалии vs норму
    mask = df.index.isin(pd.concat([low_df, high_df]).index)
    df_plot = df.copy()
    df_plot['Статус'] = np.where(mask, 'Аномалия', 'Норма')

    st.subheader("Сравнение: все позиции vs аномалии")
    fig_cmp = px.scatter(
        df_plot,
        x='Списания %', y='Закрытие потребности %',
        color='Статус',
        size='Продажа с ЗЦ сумма',
        opacity=0.6,
        hover_data=['Name_tov','Группа'],
        color_discrete_map={'Норма':'lightgrey','Аномалия':'crimson'},
        title="Аномалии выделены красным – остальные скопом серым"
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

    # 6. Экспорт
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as w:
        low_df.to_excel(w, sheet_name='Низкие', index=False)
        high_df.to_excel(w, sheet_name='Высокие', index=False)
    buf.seek(0)
    st.download_button("Скачать результат в Excel", buf,
                       "anomalies_final.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

if __name__ == "__main__":
    main()
