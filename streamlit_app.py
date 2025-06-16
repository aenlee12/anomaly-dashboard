import streamlit as st
import pandas as pd
from io import BytesIO
import plotly.express as px

st.set_page_config(page_title="Аномалии: списания и закрытие потребности", layout="wide")
st.title("Анализ аномалий: Списания и Закрытие потребности")

# 1. Загрузка CSV
uploaded_file = st.file_uploader("Загрузите CSV-файл с данными", type=["csv"])
if not uploaded_file:
    st.info("Пожалуйста, загрузите CSV-файл для анализа.")
    st.stop()

# 2. Чтение и конвертация
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
df_clean = df.dropna(subset=['Списания_%','Закрытие_потребности_%'])

# 3. Фильтры
low_waste_min, low_waste_max = 0.5, 8
low_fill_min,  low_fill_max  = 10, 75
high_waste_min, high_fill_min = 20, 80

low_anomalies  = df_clean[
    df_clean['Списания_%'].between(low_waste_min, low_waste_max) &
    df_clean['Закрытие_потребности_%'].between(low_fill_min, low_fill_max)
]
high_anomalies = df_clean[
    (df_clean['Списания_%'] >= high_waste_min) &
    (df_clean['Закрытие_потребности_%'] >= high_fill_min)
]

cols_short = ['Категория','Группа','Name_tov','Списания_%','Закрытие_потребности_%']
low_short  = low_anomalies[cols_short]
high_short = high_anomalies[cols_short]

# 4. Метрики
st.subheader("Общие метрики")
c1, c2 = st.columns(2)
c1.metric("Низкие аномалии", len(low_anomalies))
c2.metric("Высокие аномалии", len(high_anomalies))

# 5. Таблицы с форматированием и заливкой
st.subheader("Низкие аномалии")
st.dataframe(
    low_short.style
        .format({"Списания_%":"{:.1f}", "Закрытие_потребности_%":"{:.1f}"})
        .background_gradient(subset=['Списания_%'], cmap='Greens', vmin=low_waste_min, vmax=low_waste_max)
        .background_gradient(subset=['Закрытие_потребности_%'], cmap='Greens', vmin=low_fill_min, vmax=low_fill_max),
    use_container_width=True
)

st.subheader("Высокие аномалии")
st.dataframe(
    high_short.style
        .format({"Списания_%":"{:.1f}", "Закрытие_потребности_%":"{:.1f}"})
        .background_gradient(subset=['Списания_%'], cmap='Reds', vmin=high_waste_min)
        .background_gradient(subset=['Закрытие_потребности_%'], cmap='Reds', vmin=high_fill_min),
    use_container_width=True
)

# 6. Донат-диаграмма: весь объём + 2 вида аномалий
st.subheader("Доля позиций: нормальные и аномалии")
total = len(df_clean)
counts = {
    "Низкие аномалии": len(low_anomalies),
    "Высокие аномалии": len(high_anomalies),
    "Остальные позиции": total - len(low_anomalies) - len(high_anomalies)
}
fig_donut = px.pie(
    names=list(counts.keys()),
    values=list(counts.values()),
    title="Распределение по типам позиций",
    hole=0.4
)
st.plotly_chart(fig_donut, use_container_width=True)

# 7. Скачать Excel
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
