import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import plotly.express as px

def weighed_avg(series, weights):
    return np.average(series, weights=weights) if weights.sum() > 0 else np.nan

st.set_page_config(page_title="Аномалии: списания и закрытие потребности", layout="wide")
st.title("Анализ аномалий: Списания и Закрытие потребности")

uploaded_file = st.file_uploader("Загрузите CSV-файл с данными", type=["csv"])
if not uploaded_file:
    st.info("Пожалуйста, загрузите CSV-файл для анализа.")
    st.stop()

# Чтение и подготовка данных
df = pd.read_csv(uploaded_file)
df.columns = df.columns.str.strip()
df['Списания_%'] = (
    df['ЗЦ2_срок_качество_%'].astype(str)
      .str.replace(',', '.').str.rstrip('%').astype(float)
)
df['Закрытие_потребности_%'] = (
    df['Закрытие потребности_%'].astype(str)
      .str.replace(',', '.').str.rstrip('%').astype(float)
)
df['Нужно'] = pd.to_numeric(df['Нужно'], errors='coerce')
df_clean = df.dropna(subset=['Списания_%', 'Закрытие_потребности_%'])

# Пороговые фильтры
low_waste_min, low_waste_max = 0.5, 8
low_fill_min, low_fill_max   = 10, 75
high_waste_min, high_fill_min = 20, 80

low_anomalies = df_clean[
    df_clean['Списания_%'].between(low_waste_min, low_waste_max) &
    df_clean['Закрытие_потребности_%'].between(low_fill_min, low_fill_max)
]
high_anomalies = df_clean[
    (df_clean['Списания_%'] >= high_waste_min) &
    (df_clean['Закрытие_потребности_%'] >= high_fill_min)
]

cols_short = ['Категория','Группа','Name_tov','Списания_%','Закрытие_потребности_%','Нужно']
low_short  = low_anomalies[cols_short]
high_short = high_anomalies[cols_short]

# Функция для форматирования числа
fmt = "{:.1f}"

# --- Низкие аномалии ---
st.subheader("Низкие списания + Низкое закрытие потребности")

# Вычисление средних
mean_waste_low = low_anomalies['Списания_%'].mean()
mean_fill_low  = low_anomalies['Закрытие_потребности_%'].mean()
wavg_waste_low = weighed_avg(low_anomalies['Списания_%'], low_anomalies['Нужно'].fillna(0))
wavg_fill_low  = weighed_avg(low_anomalies['Закрытие_потребности_%'], low_anomalies['Нужно'].fillna(0))

# Показ метрик
c1, c2, c3, c4 = st.columns(4)
c1.metric("Среднее списания", fmt.format(mean_waste_low)+"%")
c2.metric("Взвешенное среднее списания", fmt.format(wavg_waste_low)+"%")
c3.metric("Среднее закрытие", fmt.format(mean_fill_low)+"%")
c4.metric("Взвешенное среднее закрытие", fmt.format(wavg_fill_low)+"%")

# Таблица деталей
st.dataframe(
    low_short.style
        .format({"Списания_%": fmt, "Закрытие_потребности_%": fmt, "Нужно": "{:.0f}"})
        .background_gradient(subset=['Списания_%'], cmap='Greens', vmin=low_waste_min, vmax=low_waste_max)
        .background_gradient(subset=['Закрытие_потребности_%'], cmap='Greens', vmin=low_fill_min, vmax=low_fill_max),
    use_container_width=True
)

# --- Высокие аномалии ---
st.subheader("Высокие списания + Высокое закрытие потребности")

# Вычисление средних
mean_waste_high = high_anomalies['Списания_%'].mean()
mean_fill_high  = high_anomalies['Закрытие_потребности_%'].mean()
wavg_waste_high = weighed_avg(high_anomalies['Списания_%'], high_anomalies['Нужно'].fillna(0))
wavg_fill_high  = weighed_avg(high_anomalies['Закрытие_потребности_%'], high_anomalies['Нужно'].fillna(0))

# Показ метрик
c1, c2, c3, c4 = st.columns(4)
c1.metric("Среднее списания", fmt.format(mean_waste_high)+"%")
c2.metric("Взвешенное среднее списания", fmt.format(wavg_waste_high)+"%")
c3.metric("Среднее закрытие", fmt.format(mean_fill_high)+"%")
c4.metric("Взвешенное среднее закрытие", fmt.format(wavg_fill_high)+"%")

# Таблица деталей
st.dataframe(
    high_short.style
        .format({"Списания_%": fmt, "Закрытие_потребности_%": fmt, "Нужно": "{:.0f}"})
        .background_gradient(subset=['Списания_%'], cmap='Reds', vmin=high_waste_min)
        .background_gradient(subset=['Закрытие_потребности_%'], cmap='Reds', vmin=high_fill_min),
    use_container_width=True
)

# Донат-диаграмма и скачивание аналогично предыдущему примеру...

