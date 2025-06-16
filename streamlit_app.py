import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from sklearn.ensemble import IsolationForest
import plotly.express as px

def load_and_prepare(uploaded_file):
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
    df['Продажа_с_ЗЦ_сумма'] = pd.to_numeric(df['Продажа_с_ЗЦ_сумма'], errors='coerce').fillna(0)
    return df.dropna(subset=['Списания_%', 'Закрытие_потребности_%'])

def compute_scores(df):
    X = df[['Списания_%', 'Закрытие_потребности_%']]
    iso = IsolationForest(contamination=0.05, random_state=42)
    iso.fit(X)
    df['anomaly_score'] = iso.decision_function(X)
    df['anomaly_severity'] = -df['anomaly_score']  # higher = more anomalous
    df['combined_score'] = df['anomaly_severity'] * df['Продажа_с_ЗЦ_сумма']
    return df

def filter_and_sort(df, condition):
    df_f = df[condition].copy()
    return df_f.sort_values(by=['combined_score'], ascending=False)

def display_section(df_sec, title):
    low_waste_min, low_waste_max = 0.5, 8
    low_fill_min, low_fill_max   = 10, 75
    high_waste_min, high_fill_min = 20, 80

    # Metrics
    mean_w = df_sec['Списания_%'].mean()
    wavg_w = np.average(df_sec['Списания_%'], weights=df_sec['Продажа_с_ЗЦ_сумма']) if df_sec['Продажа_с_ЗЦ_сумма'].sum() else np.nan
    mean_f = df_sec['Закрытие_потребности_%'].mean()
    wavg_f = np.average(df_sec['Закрытие_потребности_%'], weights=df_sec['Продажа_с_ЗЦ_сумма']) if df_sec['Продажа_с_ЗЦ_сумма'].sum() else np.nan

    st.subheader(title)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Среднее списания", f"{mean_w:.1f}%")
    c2.metric("Взв. среднее списания", f"{wavg_w:.1f}%")
    c3.metric("Среднее закрытие", f"{mean_f:.1f}%")
    c4.metric("Взв. среднее закрытие", f"{wavg_f:.1f}%")

    cols_disp = [
        'Категория','Группа','Name_tov',
        'Списания_%','Закрытие_потребности_%',
        'Продажа_с_ЗЦ_сумма','anomaly_severity','combined_score'
    ]
    st.dataframe(df_sec[cols_disp].style
        .format({
            'Списания_%': '{:.1f}', 'Закрытие_потребности_%': '{:.1f}',
            'Продажа_с_ЗЦ_сумма': '{:.0f}',
            'anomaly_severity': '{:.3f}', 'combined_score': '{:.2f}'
        }), use_container_width=True)

def main():
    st.set_page_config(page_title="Аномалии App", layout="wide")
    st.title("Анализ аномалий с комбинированным скором")

    uf = st.file_uploader("Загрузите CSV", type=["csv"])
    if not uf:
        st.info("Загрузите файл для старта")
        return

    df = load_and_prepare(uf)
    df = compute_scores(df)

    low_cond = df['Списания_%'].between(0.5, 8) & df['Закрытие_потребности_%'].between(10, 75)
    high_cond = df['Списания_%'] >= 20 & df['Закрытие_потребности_%'] >= 80

    low_df = filter_and_sort(df, low_cond)
    high_df = filter_and_sort(df, high_cond)

    display_section(low_df, "Низкие списания + Низкое закрытие потребности")
    display_section(high_df, "Высокие списания + Высокое закрытие потребности")

    # Donut
    total = len(df)
    counts = {
        "Низкие": len(low_df),
        "Высокие": len(high_df),
        "Остальные": total - len(low_df) - len(high_df)
    }
    fig = px.pie(names=list(counts.keys()), values=list(counts.values()), hole=0.4)
    st.plotly_chart(fig, use_container_width=True)

    # Download
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as w:
        low_df.to_excel(w, sheet_name='low', index=False)
        high_df.to_excel(w, sheet_name='high', index=False)
    buf.seek(0)
    st.download_button("Скачать XLSX", buf, "results.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

if __name__=="__main__":
    main()
