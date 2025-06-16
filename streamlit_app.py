import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import plotly.express as px
from sklearn.ensemble import IsolationForest

def load_data(uploaded):
    df = pd.read_csv(uploaded)
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

def compute_scores(df):
    X = df[['Списания %', 'Закрытие потребности %']]
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(X)
    df['Оценка аномалии'] = -model.decision_function(X)  # higher = more anomalous
    df['Комбинированный скор'] = df['Оценка аномалии'] * df['Продажа с ЗЦ сумма']
    return df

def filter_sort(df, cond):
    return df[cond].sort_values('Комбинированный скор', ascending=False)

def display_section(df_sec, title):
    # Metrics
    mean_waste = df_sec['Списания %'].mean()
    wavg_waste = np.average(df_sec['Списания %'], weights=df_sec['Продажа с ЗЦ сумма']) if df_sec['Продажа с ЗЦ сумма'].sum() else np.nan
    mean_fill = df_sec['Закрытие потребности %'].mean()
    wavg_fill = np.average(df_sec['Закрытие потребности %'], weights=df_sec['Продажа с ЗЦ сумма']) if df_sec['Продажа с ЗЦ сумма'].sum() else np.nan

    st.subheader(title)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Среднее списания %", f"{mean_waste:.1f}")
    c2.metric("Взв. среднее списания %", f"{wavg_waste:.1f}")
    c3.metric("Среднее закрытие %", f"{mean_fill:.1f}")
    c4.metric("Взв. среднее закрытие %", f"{wavg_fill:.1f}")

    cols = ['Категория', 'Группа', 'Name_tov',
            'Списания %', 'Закрытие потребности %',
            'Продажа с ЗЦ сумма', 'Оценка аномалии', 'Комбинированный скор']
    st.dataframe(
        df_sec[cols].style
            .format({
                'Списания %': '{:.1f}', 'Закрытие потребности %': '{:.1f}',
                'Продажа с ЗЦ сумма': '{:.0f}',
                'Оценка аномалии': '{:.3f}', 'Комбинированный скор': '{:.2f}'
            }),
        use_container_width=True
    )

def main():
    st.set_page_config(page_title="Аномалии", layout="wide")
    st.title("Анализ аномалий: списания и закрытие потребности")

    uploaded = st.file_uploader("Загрузите CSV", type="csv")
    if not uploaded:
        st.info("Загрузите файл для анализа")
        return

    df = load_data(uploaded)
    df = compute_scores(df)

    # conditions
    low_cond = (df['Списания %'].between(0.5, 8)) & (df['Закрытие потребности %'].between(10, 75))
    high_cond = (df['Списания %'] >= 20) & (df['Закрытие потребности %'] >= 80)

    low_df = filter_sort(df, low_cond)
    high_df = filter_sort(df, high_cond)

    display_section(low_df, "Низкие списания + Низкое закрытие потребности")
    display_section(high_df, "Высокие списания + Высокое закрытие потребности")

    # donut
    total = len(df)
    counts = {
        "Низкие": len(low_df),
        "Высокие": len(high_df),
        "Остальные": total - len(low_df) - len(high_df)
    }
    fig = px.pie(names=list(counts.keys()), values=list(counts.values()), hole=0.4)
    st.plotly_chart(fig, use_container_width=True)

    # download
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        low_df.to_excel(writer, sheet_name='Низкие аномалии', index=False)
        high_df.to_excel(writer, sheet_name='Высокие аномалии', index=False)
    buf.seek(0)
    st.download_button(
        "Скачать XLSX",
        buf,
        "anomalies.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

if __name__ == "__main__":
    main()
