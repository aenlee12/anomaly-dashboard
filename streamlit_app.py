import streamlit as st
import pandas as pd
from io import BytesIO
import plotly.express as px

# Функция для расчёта метрик
def compute_metrics(df, label):
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
        'Корреляция': df[['Списания_%', 'Закрытие_потребности_%']].corr().iloc[0, 1]
    }
    return metrics

st.set_page_config(page_title="Аномалии: списания и закрытие потребности", layout="wide")
st.title("Анализ аномалий: Списания и Закрытие потребности")

uploaded_file = st.file_uploader("Загрузите CSV-файл с данными", type=["csv"])
if not uploaded_file:
    st.info("Пожалуйста, загрузите CSV-файл для анализа.")
    st.stop()

# Чтение и преобразование
df = pd.read_csv(uploaded_file)
df['Списания_%'] = (
    df['ЗЦ2_срок_качество_%'].astype(str)
      .str.replace(',', '.')
      .str.rstrip('%')
      .astype(float)
)
df['Закрытие_потребности_%'] = (
    df['Закрытие потребности_%'].astype(str)
      .str.replace(',', '.')
      .str.rstrip('%')
      .astype(float)
)
df_clean = df.dropna(subset=['Списания_%', 'Закрытие_потребности_%'])

# Пороговые значения
low_waste_min, low_waste_max = 0.5, 8
low_fill_min,  low_fill_max  = 10, 75
high_waste_min, high_fill_min = 20, 80

# Фильтрация аномалий
low_anomalies  = df_clean[
    df_clean['Списания_%'].between(low_waste_min, low_waste_max) &
    df_clean['Закрытие_потребности_%'].between(low_fill_min, low_fill_max)
]
high_anomalies = df_clean[
    (df_clean['Списания_%'] >= high_waste_min) &
    (df_clean['Закрытие_потребности_%'] >= high_fill_min)
]

# Расчёт метрик
metrics_low = compute_metrics(low_anomalies, 'Низкие списания + Низкое закрытие потребности')
metrics_high = compute_metrics(high_anomalies, 'Высокие списания + Высокое закрытие потребности')
metrics_df = pd.DataFrame([metrics_low, metrics_high])

# Вывод метрик
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
    'Корреляция': '{:.2f}'
}))

# Колонки коротких таблиц
cols_short = ['Категория', 'Группа', 'Name_tov', 'Списания_%', 'Закрытие_потребности_%']
low_short  = low_anomalies[cols_short]
high_short = high_anomalies[cols_short]

# Вывод стилизованных таблиц
st.subheader("Низкие списания + Низкое закрытие потребности")
st.dataframe(
    low_short.style
        .format({"Списания_%": "{:.1f}", "Закрытие_потребности_%": "{:.1f}"})
        .background_gradient(subset=['Списания_%'], cmap='Greens', vmin=low_waste_min, vmax=low_waste_max)
        .background_gradient(subset=['Закрытие_потребности_%'], cmap='Greens', vmin=low_fill_min, vmax=low_fill_max),
    use_container_width=True
)

st.subheader("Высокие списания + Высокое закрытие потребности")
st.dataframe(
    high_short.style
        .format({"Списания_%": "{:.1f}", "Закрытие_потребности_%": "{:.1f}"})
        .background_gradient(subset=['Списания_%'], cmap='Reds', vmin=high_waste_min)
        .background_gradient(subset=['Закрытие_потребности_%'], cmap='Reds', vmin=high_fill_min),
    use_container_width=True
)

# Донат-диаграмма
st.subheader("Доля типов позиций")
total = len(df_clean)
counts = {
    "Низкие списания + Низкое закрытие потребности": len(low_anomalies),
    "Высокие списания + Высокое закрытие потребности": len(high_anomalies),
    "Остальные позиции": total - len(low_anomalies) - len(high_anomalies)
}
fig_donut = px.pie(
    names=list(counts.keys()),
    values=list(counts.values()),
    title="Распределение позиций",
    hole=0.4
)
st.plotly_chart(fig_donut, use_container_width=True)

# Кнопка скачивания результатов
buffer = BytesIO()
with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
    low_anomalies.to_excel(writer, sheet_name='low_anomalies', index=False)
    high_anomalies.to_excel(writer, sheet_name='high_anomalies', index=False)
    low_short.to_excel(writer, sheet_name='low_short', index=False)
    high_short.to_excel(writer, sheet_name='high_short', index=False)
buffer.seek(0)
st.download_button(
    "Скачать результаты в Excel",
    data=buffer,
    file_name="anomalies.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
