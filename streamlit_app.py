import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import plotly.express as px
from sklearn.ensemble import IsolationForest

def main():
    st.set_page_config(page_title="Аномалии: списания и закрытие потребности", layout="wide")
    st.title("Анализ аномалий: Списания и Закрытие потребности")

    uploaded_file = st.file_uploader("Загрузите CSV-файл с данными", type=["csv"])
    if not uploaded_file:
        st.info("Пожалуйста, загрузите CSV-файл для анализа.")
        return

    # Чтение и подготовка данных
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()
    # Конвертация строковых процентов
    df['Списания_%'] = (
        df['ЗЦ2_срок_качество_%'].astype(str)
        .str.replace(',', '.').str.rstrip('%').astype(float)
    )
    df['Закрытие_потребности_%'] = (
        df['Закрытие потребности_%'].astype(str)
        .str.replace(',', '.').str.rstrip('%').astype(float)
    )
    # Ensure sales column exists
    df['Продажа_с_ЗЦ_сумма'] = pd.to_numeric(df['Продажа_с_ЗЦ_сумма'], errors='coerce').fillna(0)
    df_clean = df.dropna(subset=['Списания_%', 'Закрытие_потребности_%'])

    # Обучение IsolationForest для оценки аномалий
    X = df_clean[['Списания_%', 'Закрытие_потребности_%']]
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(X)
    df_clean['anomaly_score'] = model.decision_function(X)
    df_clean['is_anomaly'] = model.predict(X) == -1

    # Пороговые фильтры
    low_waste_min, low_waste_max = 0.5, 8
    low_fill_min, low_fill_max   = 10, 75
    high_waste_min, high_fill_min = 20, 80

    # Выборки аномалий
    low_anomalies = df_clean[
        df_clean['Списания_%'].between(low_waste_min, low_waste_max) &
        df_clean['Закрытие_потребности_%'].between(low_fill_min, low_fill_max)
    ]
    high_anomalies = df_clean[
        (df_clean['Списания_%'] >= high_waste_min) &
        (df_clean['Закрытие_потребности_%'] >= high_fill_min)
    ]

    # Краткие таблицы
    cols_display = ['Категория', 'Группа', 'Name_tov', 
                    'Списания_%', 'Закрытие_потребности_%', 
                    'Продажа_с_ЗЦ_сумма', 'anomaly_score']
    # Сортируем сначала по аномальности (score ASC), потом по продажам DESC
    low_short = low_anomalies[cols_display] \
        .sort_values(by=['anomaly_score', 'Продажа_с_ЗЦ_сумма'], ascending=[True, False])
    high_short = high_anomalies[cols_display] \
        .sort_values(by=['anomaly_score', 'Продажа_с_ЗЦ_сумма'], ascending=[True, False])

    # Показ метрик
    st.subheader("Метрики и сортировка по аномалии и продажам")
    c1, c2 = st.columns(2)
    c1.metric("Низкие списания + Низкое закрытие", len(low_short))
    c2.metric("Высокие списания + Высокое закрытие", len(high_short))

    # Таблицы
    st.subheader("Низкие списания + Низкое закрытие — детали")
    st.dataframe(low_short, use_container_width=True)

    st.subheader("Высокие списания + Высокое закрытие — детали")
    st.dataframe(high_short, use_container_width=True)

    # Донат-диаграмма
    total = len(df_clean)
    counts = {
        "Низкие списания + Низкое закрытие": len(low_short),
        "Высокие списания + Высокое закрытие": len(high_short),
        "Остальные": total - len(low_short) - len(high_short)
    }
    fig_donut = px.pie(
        names=list(counts.keys()), values=list(counts.values()),
        title="Распределение позиций", hole=0.4
    )
    st.plotly_chart(fig_donut, use_container_width=True)

    # Скачать Excel
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        low_anomalies.to_excel(writer, sheet_name='low_anomalies', index=False)
        high_anomalies.to_excel(writer, sheet_name='high_anomalies', index=False)
        low_short.to_excel(writer, sheet_name='low_short', index=False)
        high_short.to_excel(writer, sheet_name='high_short', index=False)
    buffer.seek(0)
    st.download_button(
        "Скачать результаты в Excel",
        data=buffer, file_name="anomalies.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

if __name__ == "__main__":
    main()
