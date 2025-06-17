import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from sklearn.ensemble import IsolationForest

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
    df['anomaly_score'] = -iso.decision_function(X)
    df['combined_score'] = df['anomaly_score'] * df['Продажа с ЗЦ сумма']
    return df

def main():
    st.title("Анализ аномалий: списания и закрытие потребности")

    uploaded = st.file_uploader("Загрузите CSV", type="csv")
    if not uploaded:
        st.info("Пожалуйста, загрузите CSV для анализа")
        return

    df = load_and_prepare(uploaded)
    df = score_anomalies(df)

    # Определяем глобальные min/max для продаж
    sale_min, sale_max = float(df['Продажа с ЗЦ сумма'].min()), float(df['Продажа с ЗЦ сумма'].max())

    # --- Пресеты ---
    st.sidebar.header("Горячие пресеты")
    if st.sidebar.button("Слабая чувствительность"):
        st.session_state.sale_range = (sale_min, sale_max)
        st.session_state.low_waste = (0.5, 15.0)
        st.session_state.low_fill  = (5.0, 85.0)
        st.session_state.high_waste = 15.0
        st.session_state.high_fill  = 60.0

    if st.sidebar.button("Средняя чувствительность"):
        st.session_state.sale_range = (sale_min, sale_max)
        st.session_state.low_waste = (0.5, 8.0)
        st.session_state.low_fill  = (10.0, 75.0)
        st.session_state.high_waste = 20.0
        st.session_state.high_fill  = 80.0

    if st.sidebar.button("Высокая чувствительность"):
        st.session_state.sale_range = (sale_min, sale_max)
        st.session_state.low_waste = (0.5, 5.0)
        st.session_state.low_fill  = (20.0, 60.0)
        st.session_state.high_waste = 25.0
        st.session_state.high_fill  = 90.0

    # --- Слайдеры (с инициализацией из session_state) ---
    st.sidebar.header("Настройки фильтрации")
    sale_range = st.sidebar.slider(
        "Сумма продаж (руб.)",
        sale_min, sale_max,
        tuple(st.session_state.get('sale_range', (sale_min, sale_max))),
        key='sale_range'
    )
    low_waste = st.sidebar.slider(
        "Низкие списания % диапазон",
        0.0, 100.0,
        tuple(st.session_state.get('low_waste', (0.5, 8.0))),
        key='low_waste'
    )
    low_fill = st.sidebar.slider(
        "Низкое закрытие % диапазон",
        0.0, 100.0,
        tuple(st.session_state.get('low_fill', (10.0, 75.0))),
        key='low_fill'
    )
    high_waste = st.sidebar.slider(
        "Высокие списания % порог",
        0.0, 200.0,
        float(st.session_state.get('high_waste', 20.0)),
        key='high_waste'
    )
    high_fill = st.sidebar.slider(
        "Высокое закрытие % порог",
        0.0, 200.0,
        float(st.session_state.get('high_fill', 80.0)),
        key='high_fill'
    )

    # --- Фильтрация данных ---
    df = df[
        (df['Продажа с ЗЦ сумма'] >= sale_range[0]) &
        (df['Продажа с ЗЦ сумма'] <= sale_range[1])
    ]
    low_df = df[
        df['Списания %'].between(*low_waste) &
        df['Закрытие потребности %'].between(*low_fill)
    ].sort_values('combined_score', ascending=False)
    high_df = df[
        (df['Списания %'] >= high_waste) &
        (df['Закрытие потребности %'] >= high_fill)
    ].sort_values('combined_score', ascending=False)

    # --- Вывод таблиц с заливкой ---
    def display_table(ddf, title):
        st.subheader(f"{title} (найдено {len(ddf)})")
        cols = [
            'Категория','Группа','Name_tov',
            'Списания %','Закрытие потребности %',
            'Продажа с ЗЦ сумма','anomaly_score','combined_score'
        ]
        styler = ddf[cols].style.format({
            'Списания %': '{:.1f}',
            'Закрытие потребности %': '{:.1f}',
            'Продажа с ЗЦ сумма': '{:.0f}',
            'anomaly_score': '{:.3f}',
            'combined_score': '{:.0f}',
        })
        styler = (
            styler
            .background_gradient(subset=['Списания %'], cmap='Reds')
            .background_gradient(subset=['Закрытие потребности %'], cmap='Blues')
            .background_gradient(subset=['combined_score'], cmap='Purples')
        )
        st.dataframe(styler, use_container_width=True)

    display_table(low_df,  "Низкие списания + низкое закрытие потребности")
    display_table(high_df, "Высокие списания + высокое закрытие потребности")

    # --- Скачивание в Excel ---
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        low_df.to_excel(writer, sheet_name='Низкие', index=False)
        high_df.to_excel(writer, sheet_name='Высокие', index=False)
    buf.seek(0)
    st.download_button(
        "Скачать результаты в Excel",
        data=buf,
        file_name="anomalies.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

if __name__ == "__main__":
    main()
