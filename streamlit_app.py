import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import plotly.express as px

def compute_metrics(df, label):
    # Обычные метрики
    metrics = {
        'Тип аномалии': label,
        'Количество': len(df),
        'Среднее_списания_%': df['Списания_%'].mean(),
        'Медианное_списания_%': df['Списания_%'].median(),
        'Минимальное_списания_%': df['Списания_%'].min(),
        'Максимальное_списания_%': df['Списания_%'].max(),
        'Среднее_закрытие_%': df['Закрытие_потребности_%'].mean(),
        'Медианное_закрытие_%': df['Закрытие_потребности_%'].median(),
        'Минимальное_закрытие_%': df['Закрытие_потребности_%'].min(),
        'Максимальное_закрытие_%': df['Закрытие_потребности_%'].max(),
        'Корреляция': df[['Списания_%','Закрытие_потребности_%']].corr().iloc[0,1]
    }
    # Взвешенное среднее: вес = колонка «Нужно»
    w = df['Нужно'].fillna(0).astype(float)
    # Защита от деления на ноль
    if w.sum() > 0:
        metrics['Взвешенное_среднее_списания_%'] = np.average(df['Списания_%'], weights=w)
        metrics['Взвешенное_среднее_закрытие_%'] = np.average(df['Закрытие_потребности_%'], weights=w)
    else:
        metrics['Взвешенное_среднее_списания_%'] = np.nan
        metrics['Взвешенное_среднее_закрытие_%'] = np.nan
    return metrics

st.set_page_config(page_title="Аномалии: списания и закрытие потребности", layout="wide")
st.title("Анализ аномалий: Списания и Закрытие потребности")

uploaded_file = st.file_uploader("Загрузите CSV-файл с данными", type=["csv"])
if not uploaded_file:
    st.info("Пожалуйста, загрузите CSV-файл для анализа.")
    st.stop()

# ———————————————
# 1) Чтение и подготовка
df = pd.read_csv(uploaded_file)
# Убираем случайные пробелы в названиях столбцов
df.columns = df.columns.str.strip()

# Конвертация строковых процентов
df['Списания_%'] = (
    df['ЗЦ2_срок_качество_%']
     .astype(str).str.replace(',', '.').str.rstrip('%').astype(float)
)
df['Закрытие_потребности_%'] = (
    df['Закрытие потребности_%']
     .astype(str).str.replace(',', '.').str.rstrip('%').astype(float)
)

# Убедимся, что колонка «Нужно» есть и числовая
df['Нужно'] = pd.to_numeric(df['Нужно'], errors='coerce')

df_clean = df.dropna(subset=['Списания_%','Закрытие_потребности_%'])

# ———————————————
# 2) Пороговые фильтры
low_waste_min, low_waste_max = 0.5, 8
low_fill_min,  low_fill_max  = 10, 75
high_waste_min, high_fill_min = 20, 80

low_anomalies = df_clean[
    df_clean['Списания_%'].between(low_waste_min, low_waste_max) &
    df_clean['Закрытие_потребности_%'].between(low_fill_min, low_fill_max)
]
high_anomalies = df_clean[
    (df_clean['Списания_%'] >= high_waste_min) &
    (df_clean['Закрытие_потребности_%'] >= high_fill_min)
]

# ———————————————
# 3) Сбор метрик
metrics_low  = compute_metrics(low_anomalies,  'Низкие списания + Низкое закрытие потребности')
metrics_high = compute_metrics(high_anomalies, 'Высокие списания + Высокое закрытие потребности')
metrics_df   = pd.DataFrame([metrics_low, metrics_high])

# ———————————————
# 4) Вывод сводных метрик
st.subheader("Сводные метрики по типам аномалий")
st.table(metrics_df.style.format({
    'Среднее_списания_%': '{:.1f}',
    'Медианное_списания_%': '{:.1f}',
    'Минимальное_списания_%': '{:.1f}',
    'Максимальное_списания_%': '{:.1f}',
    'Среднее_закрытие_%': '{:.1f}',
    'Медианное_закрытие_%': '{:.1f}',
    'Минимальное_закрытие_%': '{:.1f}',
    'Максимальное_закрытие_%': '{:.1f}',
    'Взвешенное_среднее_списания_%': '{:.1f}',
    'Взвешенное_среднее_закрытие_%': '{:.1f}',
    'Корреляция': '{:.2f}'
}))

# ———————————————
# 5) Таблицы по позициям (кратко)
cols_short = ['Категория','Группа','Name_tov','Списания_%','Закрытие_потребности_%','Нужно']
low_short  = low_anomalies[cols_short]
high_short = high_anomalies[cols_short]

st.subheader("Низкие списания + Низкое закрытие потребности — детали")
st.dataframe(
    low_short.style
        .format({"Списания_%":"{:.1f}", "Закрытие_потребности_%":"{:.1f}", "Нужно":"{:.0f}"})
        .background_gradient(subset=['Списания_%'], cmap='Greens', vmin=low_waste_min, vmax=low_waste_max)
        .background_gradient(subset=['Закрытие_потребности_%'], cmap='Greens', vmin=low_fill_min, vmax=low_fill_max),
    use_container_width=True
)

st.subheader("Высокие списания + Высокое закрытие потребности — детали")
st.dataframe(
    high_short.style
        .format({"Списания_%":"{:.1f}", "Закрытие_потребности_%":"{:.1f}", "Нужно":"{:.0f}"})
        .background_gradient(subset=['Списания_%'], cmap='Reds', vmin=high_waste_min)
        .background_gradient(subset=['Закрытие_потребности_%'], cmap='Reds', vmin=high_fill_min),
    use_container_width=True
)

# ———————————————
# 6) Донат-диаграмма и скачивание
# (остальные участки кода без изменений)
