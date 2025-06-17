import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="Аномалии: списания & закрытие", layout="wide")

# Initialize session state for presets
if 'sale_range' not in st.session_state:
    st.session_state.sale_range = None
if 'low_waste' not in st.session_state:
    st.session_state.low_waste = None
if 'low_fill' not in st.session_state:
    st.session_state.low_fill = None
if 'high_waste' not in st.session_state:
    st.session_state.high_waste = None
if 'high_fill' not in st.session_state:
    st.session_state.high_fill = None

@st.cache_data
def load_and_prepare(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df['Списания %'] = pd.to_numeric(df['ЗЦ2_срок_качество_%'].str.replace(',', '.').str.rstrip('%'), errors='coerce')
    df['Закрытие потребности %'] = pd.to_numeric(df['Закрытие потребности_%'].str.replace(',', '.').str.rstrip('%'), errors='coerce')
    df['Продажа с ЗЦ сумма'] = pd.to_numeric(df['Продажа_с_ЗЦ_сумма'], errors='coerce').fillna(0)
    return df.dropna(subset=['Списания %','Закрытие потребности %'])

@st.cache_data
def score_anomalies(df):
    X = df[['Списания %','Закрытие потребности %']]
    iso = IsolationForest(contamination=0.05, random_state=42)
    iso.fit(X)
    df['anomaly_score'] = -iso.decision_function(X)
    df['combined_score'] = df['anomaly_score'] * df['Продажа с ЗЦ сумма']
    return df

def display_anomaly_table(df, title):
    st.subheader(f"{title} (найдено {len(df)})")
    cols = ['Категория','Группа','Name_tov',
            'Списания %','Закрытие потребности %',
            'Продажа с ЗЦ сумма','anomaly_score','combined_score']
    styler = df[cols].style.format({
        'Списания %': '{:.1f}',
        'Закрытие потребности %': '{:.1f}',
        'Продажа с ЗЦ сумма': '{:.0f}',
        'anomaly_score': '{:.3f}',
        'combined_score': '{:.0f}',
    }).background_gradient(subset=['Списания %'], cmap='Reds') \
      .background_gradient(subset=['Закрытие потребности %'], cmap='Blues') \
      .background_gradient(subset=['combined_score'], cmap='Purples')
    st.dataframe(styler, use_container_width=True)

def apply_presets(preset):
    # Define presets
    presets = {
        'Слабая чувствительность': {
            'sale_range': (sale_min, sale_max),
            'low_waste': (0.5, 15.0),
            'low_fill': (5.0, 85.0),
            'high_waste': 15.0,
            'high_fill': 60.0
        },
        'Средняя чувствительность': {
            'sale_range': (sale_min, sale_max),
            'low_waste': (0.5, 8.0),
            'low_fill': (10.0, 75.0),
            'high_waste': 20.0,
            'high_fill': 80.0
        },
        'Высокая чувствительность': {
            'sale_range': (sale_min, sale_max),
            'low_waste': (0.5, 5.0),
            'low_fill': (20.0, 60.0),
            'high_waste': 25.0,
            'high_fill': 90.0
        }
    }
    p = presets[preset]
    st.session_state.sale_range = p['sale_range']
    st.session_state.low_waste = p['low_waste']
    st.session_state.low_fill = p['low_fill']
    st.session_state.high_waste = p['high_waste']
    st.session_state.high_fill = p['high_fill']

def main():
    st.title("Анализ аномалий: списания и закрытие потребности")

    uploaded = st.file_uploader("Загрузите CSV-файл", type="csv")
    if not uploaded:
        st.info("Пожалуйста, загрузите CSV для анализа")
        return

    df = load_and_prepare(uploaded)
    df = score_anomalies(df)

    # Sidebar filters and presets
    st.sidebar.header("Настройки фильтрации")
    global sale_min, sale_max
    sale_min, sale_max = float(df['Продажа с ЗЦ сумма'].min()), float(df['Продажа с ЗЦ сумма'].max())

    sale_range = st.sidebar.slider("Сумма продаж (руб.)",
                                   sale_min, sale_max,
                                   st.session_state.sale_range or (sale_min, sale_max),
                                   key='sale_range')
    low_waste = st.sidebar.slider("Низкие списания % диапазон",
                                  0.0, 100.0,
                                  st.session_state.low_waste or (0.5, 8.0),
                                  key='low_waste')
    low_fill = st.sidebar.slider("Низ. закрытие % диапазон",
                                 0.0, 100.0,
                                 st.session_state.low_fill or (10.0, 75.0),
                                 key='low_fill')
    high_waste = st.sidebar.slider("Высокие списания % порог",
                                   0.0, 200.0,
                                   st.session_state.high_waste or 20.0,
                                   key='high_waste')
    high_fill = st.sidebar.slider("Выс. закрытие % порог",
                                  0.0, 200.0,
                                  st.session_state.high_fill or 80.0,
                                  key='high_fill')

    st.sidebar.markdown("---")
    st.sidebar.write("Горячие пресеты:")
    if st.sidebar.button("Слабая чувствительность"):
        apply_presets('Слабая чувствительность')
    if st.sidebar.button("Средняя чувствительность"):
        apply_presets('Средняя чувствительность')
    if st.sidebar.button("Высокая чувствительность"):
        apply_presets('Высокая чувствительность')

    # apply filters
    df = df[(df['Продажа с ЗЦ сумма'] >= sale_range[0]) &
            (df['Продажа с ЗЦ сумма'] <= sale_range[1])]

    low_df = df[
        df['Списания %'].between(*low_waste) &
        df['Закрытие потребности %'].between(*low_fill)
    ].sort_values('combined_score', ascending=False)

    high_df = df[
        (df['Списания %'] >= high_waste) &
        (df['Закрытие потребности %'] >= high_fill)
    ].sort_values('combined_score', ascending=False)

    display_anomaly_table(low_df,  "Низкие списания + низкое закрытие потребности")
    display_anomaly_table(high_df, "Высокие списания + высокое закрытие потребности")

    # download
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

