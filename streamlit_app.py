import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from sklearn.ensemble import IsolationForest
import plotly.express as px

@st.cache_data
def load_data(uploaded):
    df = pd.read_csv(uploaded)
    df.columns = df.columns.str.strip()
    # процент списаний
    df['Списания %'] = (
        df['ЗЦ2_срок_качество_%']
        .astype(str).str.replace(',', '.').str.rstrip('%').astype(float)
    )
    # процент закрытия потребности
    df['Закрытие потребности %'] = (
        df['Закрытие потребности_%']
        .astype(str).str.replace(',', '.').str.rstrip('%').astype(float)
    )
    # сумма продаж
    df['Продажа с ЗЦ сумма'] = pd.to_numeric(
        df['Продажа_с_ЗЦ_сумма'], errors='coerce'
    ).fillna(0)
    return df.dropna(subset=['Списания %','Закрытие потребности %'])

@st.cache_data
def compute_scores(df):
    # 1) чистый скор аномалии
    X = df[['Списания %','Закрытие потребности %']]
    iso = IsolationForest(contamination=0.05, random_state=42)
    iso.fit(X)
    df['anomaly_score'] = -iso.decision_function(X)

    # 2) доля продаж в группе
    df['sales_share'] = (
        df['Продажа с ЗЦ сумма']
        / df.groupby('Группа')['Продажа с ЗЦ сумма']
              .transform('sum').replace(0, np.nan)
    ).fillna(0)

    # 3) adjusted_score
    df['adjusted_score'] = df['anomaly_score'] * df['sales_share']

    # 4) нормировка
    max_abs = df['adjusted_score'].abs().max() or 1
    df['relative_adjusted'] = df['adjusted_score'] / max_abs
    return df

def main():
    st.set_page_config(page_title="Anomaly Dashboard", layout="wide")
    st.title("Dashboard аномалий: списания & закрытие потребности")

    uploaded = st.file_uploader("Загрузите CSV", type="csv")
    if not uploaded:
        return

    df = load_data(uploaded)
    df = compute_scores(df)

    # фильтруем «мёртвые» позиции
    df = df[df['Закрытие потребности %'] > 5]

    # пользовательские пороги
    p_anom = st.sidebar.slider("Перцентиль anomaly_score", 50, 99, 90)
    p_adj  = st.sidebar.slider("Перцентиль adjusted_score", 50, 99, 95)
    top_n  = st.sidebar.number_input("Топ-N", 5, 100, 30)

    thr_anom = df['anomaly_score'].quantile(p_anom/100)
    thr_adj  = df['adjusted_score'].quantile( p_adj/100)

    # реальные аномалии
    real = df[
        (df['anomaly_score'] >= thr_anom) &
        (df['adjusted_score']  >= thr_adj)
    ].sort_values('adjusted_score', ascending=False)

    # ключевые метрики
    st.subheader("Ключевые метрики")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Найдено аномалий", len(real))
    c2.metric("Порог anomaly_score", f"{thr_anom:.3f}")
    c3.metric("Порог adjusted_score", f"{thr_adj:.4f}")
    c4.metric("Макс. доля в группе", f"{(real['sales_share'].max()*100):.1f}%")

    # топ-N таблица
    st.subheader(f"Топ-{top_n} позиций по adjusted_score")
    cols = [
        'Категория','Группа','Name_tov',
        'Списания %','Закрытие потребности %',
        'Продажа с ЗЦ сумма','sales_share',
        'anomaly_score','adjusted_score','relative_adjusted'
    ]
    top = real.head(top_n)[cols]
    styler = top.style.format({
        'Списания %': '{:.1f}',
        'Закрытие потребности %': '{:.1f}',
        'Продажа с ЗЦ сумма': '{:.0f}',
        'sales_share': '{:.2%}',
        'anomaly_score': '{:.3f}',
        'adjusted_score': '{:.4f}',
        'relative_adjusted': '{:.2%}',
    })
    # цветовая заливка
    styler = (
        styler
        .background_gradient(subset=['Списания %'], cmap='Reds')
        .background_gradient(subset=['Закрытие потребности %'], cmap='Blues')
        .background_gradient(subset=['adjusted_score'],   cmap='Purples')
    )
    st.dataframe(styler, use_container_width=True)

    # влияние по группам
    st.subheader("Влияние по группам")
    grp = (
        real.groupby('Группа')
        .agg(
            Количество=('Name_tov','count'),
            Сумма_потерь=('adjusted_score','sum'),
            Средн_anom=('anomaly_score','mean'),
            Средн_rel=('relative_adjusted','mean')
        ).reset_index()
    )
    st.dataframe(grp.style.format({
        'Сумма_потерь':'{:.4f}',
        'Средн_anom':'{:.3f}',
        'Средн_rel':'{:.2%}'
    }), use_container_width=True)

    # донат-диаграмма
    counts = {
        "Реальные аномалии": len(real),
        "Остальные": len(df)-len(real)
    }
    fig = px.pie(names=list(counts), values=list(counts.values()), hole=0.4)
    st.plotly_chart(fig, use_container_width=True)

    # скачивание
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as w:
        top.to_excel(w, sheet_name='Топ-N', index=False)
        grp.to_excel(w, sheet_name='Влияние по группам', index=False)
    buf.seek(0)
    st.download_button(
        "Скачать в Excel",
        data=buf,
        file_name="anomalies_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

if __name__ == "__main__":
    main()
