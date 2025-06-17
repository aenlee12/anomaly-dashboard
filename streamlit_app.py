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
    df['Списания %'] = pd.to_numeric(df['ЗЦ2_срок_качество_%']
                                     .str.replace(',', '.').str.rstrip('%'),
                                     errors='coerce')
    df['Закрытие потребности %'] = pd.to_numeric(df['Закрытие потребности_%']
                                                .str.replace(',', '.').str.rstrip('%'),
                                                errors='coerce')
    df['Продажа с ЗЦ сумма'] = pd.to_numeric(df['Продажа_с_ЗЦ_сумма'],
                                            errors='coerce').fillna(0)
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
    st.subheader(f"{title}  (найдено {len(df)})")
    cols = [
        'Категория','Группа','Name_tov',
        'Списания %','Закрытие потребности %',
        'Продажа с ЗЦ сумма','anomaly_score','combined_score'
    ]
    styler = df[cols].style.format({
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

def main():
    st.title("Анализ аномалий: списания и закрытие потребности")
    uploaded = st.file_uploader("Загрузите CSV-файл", type="csv")
    if not uploaded:
        return

    df = load_and_prepare(uploaded)
    df = score_anomalies(df)

    # Определяем min/max по выручке для ползунка
    sale_min, sale_max = float(df['Продажа с ЗЦ сумма'].min()), float(df['Продажа с ЗЦ сумма'].max())

    # --- Выбор пресета ---
    preset = st.sidebar.radio("Пресет чувствительности", [
        "Нет (ручная настройка)",
        "Слабая чувствительность",
        "Средняя чувствительность",
        "Высокая чувствительность"
    ])

    # Задаём default-значения подпресетов
    if preset == "Слабая чувствительность":
        sale_def = (sale_min, sale_max)
        low_waste_def = (0.5, 15.0)
        low_fill_def  = (5.0, 85.0)
        high_waste_def, high_fill_def = 15.0, 60.0
    elif preset == "Средняя чувствительность":
        sale_def = (sale_min, sale_max)
        low_waste_def = (0.5, 8.0)
        low_fill_def  = (10.0, 75.0)
        high_waste_def, high_fill_def = 20.0, 80.0
    elif preset == "Высокая чувствительность":
        sale_def = (sale_min, sale_max)
        low_waste_def = (0.5, 5.0)
        low_fill_def  = (20.0, 60.0)
        high_waste_def, high_fill_def = 25.0, 90.0
    else:
        # ручная настройка по умолчанию
        sale_def = (sale_min, sale_max)
        low_waste_def = (0.5, 8.0)
        low_fill_def  = (10.0, 75.0)
        high_waste_def, high_fill_def = 20.0, 80.0

    # --- Отображаем слайдеры с default из пресета ---
    st.sidebar.header("Настройки фильтрации")
    sale_range = st.sidebar.slider(
        "Сумма продаж (руб.)",
        sale_min, sale_max, sale_def
    )
    low_waste = st.sidebar.slider(
        "Низкие списания % диапазон",
        0.0, 100.0, low_waste_def
    )
    low_fill = st.sidebar.slider(
        "Низкое закрытие % диапазон",
        0.0, 100.0, low_fill_def
    )
    high_waste = st.sidebar.slider(
        "Высокие списания % порог",
        0.0, 200.0, high_waste_def
    )
    high_fill = st.sidebar.slider(
        "Высокое закрытие % порог",
        0.0, 200.0, high_fill_def
    )

    # --- Фильтрация по продажам и порогам ---
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

    # --- Вывод таблиц ---
    display_anomaly_table(low_df,  "Низкие списания + низкое закрытие потребности")
    display_anomaly_table(high_df, "Высокие списания + высокое закрытие потребности")

    # --- Скачать Excel ---
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
