import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from sklearn.ensemble import IsolationForest
import plotly.express as px

# 1. Загрузка и подготовка данных
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()
    df['Списания %'] = (
        df['ЗЦ2_срок_качество_%']
        .astype(str).str.replace(',', '.').str.rstrip('%').astype(float)
    )
    df['Закрытие потребности %'] = (
        df['Закрытие потребности_%']
        .astype(str).str.replace(',', '.').str.rstrip('%').astype(float)
    )
    df['Продажа с ЗЦ сумма'] = pd.to_numeric(df['Продажа_с_ЗЦ_сумма'], errors='coerce').fillna(0)
    return df.dropna(subset=['Списания %', 'Закрытие потребности %'])

# 2. Вычисление скорингов
@st.cache_data
def compute_scores(df):
    # Обучаем Isolation Forest на двух признаках
    X = df[['Списания %', 'Закрытие потребности %']]
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(X)
    scores = model.decision_function(X)
    df['anomaly_score'] = -scores  # чем выше, тем более аномально
    # Доля продаж в группе
    df['sales_share_in_group'] = (
        df['Продажа с ЗЦ сумма'] /
        df.groupby('Группа')['Продажа с ЗЦ сумма'].transform('sum')
    )
    # Комбинированный скор: аномалия × вес (продажи)
    df['combined_score'] = df['anomaly_score'] * df['Продажа с ЗЦ сумма']
    # Нормировка комбинированного скора в [0;1]
    max_cs = df['combined_score'].abs().max()
    df['relative_combined_score'] = df['combined_score'] / max_cs if max_cs != 0 else 0
    return df

# 3. Главная функция приложения
def main():
    st.set_page_config(page_title="Anomaly Dashboard", layout="wide")
    st.title("Dashboard аномалий: списания и закрытие потребности")

    uploaded = st.file_uploader("Загрузите CSV-файл", type="csv")
    if not uploaded:
        st.info("Загрузите файл для анализа")
        return

    # Загрузка и скоринг
    df = load_data(uploaded)
    df = compute_scores(df)

    # 4. Параметры фильтрации
    p_anom = st.sidebar.slider("Перцентиль anomaly_score", 50, 99, 90)
    p_comb = st.sidebar.slider("Перцентиль combined_score", 50, 99, 95)
    th_sales = st.sidebar.slider("Мин. доля продаж в группе (%)", 0.0, 20.0, 5.0) / 100
    top_n = st.sidebar.number_input("Число топ-артикулов", min_value=5, max_value=100, value=30)

    # 5. Фильтрация реальных аномалий
    thr_anom = df['anomaly_score'].quantile(p_anom/100)
    thr_comb = df['combined_score'].quantile(p_comb/100)
    is_real = (
        (df['anomaly_score'] >= thr_anom) &
        (df['sales_share_in_group'] >= th_sales) &
        (df['combined_score'] >= thr_comb)
    )
    real_anoms = df[is_real].copy().sort_values('combined_score', ascending=False)

    # 6. Метрики ключевые
    st.subheader("Основные метрики")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Найдено аномалий", len(real_anoms))
    c2.metric("Порог anomaly_score", f"{thr_anom:.3f}")
    c3.metric("Порог combined_score", f"{thr_comb:.2f}")
    c4.metric("Мин. доля продаж", f"{th_sales*100:.1f}%")

    # 7. Таблица топ-N аномалий
    st.subheader(f"Топ-{top_n} артикулов по combined_score")
    cols = ['Категория','Группа','Name_tov',
            'Списания %','Закрытие потребности %',
            'Продажа с ЗЦ сумма','sales_share_in_group',
            'anomaly_score','combined_score','relative_combined_score']
    table = real_anoms[cols].head(top_n)
    st.dataframe(table.style.format({
        'Списания %': '{:.1f}',
        'Закрытие потребности %': '{:.1f}',
        'Продажа с ЗЦ сумма': '{:.0f}',
        'sales_share_in_group': '{:.2%}',
        'anomaly_score': '{:.3f}',
        'combined_score': '{:.2f}',
        'relative_combined_score': '{:.2%}'
    }), use_container_width=True)

    # 8. Влияние по категориям
    st.subheader("Влияние по группам")
    grp = (
        real_anoms.groupby('Группа')
        .agg(Количество=('Name_tov','count'),
             Сумма_потерь=('combined_score','sum'),
             Средн_anom=('anomaly_score','mean'),
             Средн_rel_cs=('relative_combined_score','mean'))
        .reset_index()
    )
    st.dataframe(grp.style.format({
        'Сумма_потерь':'{:.2f}',
        'Средн_anom':'{:.3f}',
        'Средн_rel_cs':'{:.2%}'
    }), use_container_width=True)

    # 9. Скачивание отчёта
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        table.to_excel(writer, sheet_name='Топ N', index=False)
        grp.to_excel(writer, sheet_name='Влияние по группам', index=False)
    buf.seek(0)
    st.download_button(
        "Скачать отчёт в Excel",
        data=buf,
        file_name="anomalies_insights.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

if __name__ == "__main__":
    main()
