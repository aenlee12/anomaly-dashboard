import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="Аномалии: списания и закрытие", layout="wide")

@st.cache_data
def load_and_prepare(file):
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()
    df['Списания %'] = pd.to_numeric(
        df['ЗЦ2_срок_качество_%'].str.replace(',', '.').str.rstrip('%'),
        errors='coerce'
    )
    df['Закрытие потребности %'] = pd.to_numeric(
        df['Закрытие потребности_%'].str.replace(',', '.').str.rstrip('%'),
        errors='coerce'
    )
    df['Продажа с ЗЦ сумма'] = pd.to_numeric(df['Продажа_с_ЗЦ_сумма'], errors='coerce').fillna(0)
    return df.dropna(subset=['Списания %','Закрытие потребности %'])

@st.cache_data
def compute_anomaly(df):
    X = df[['Списания %','Закрытие потребности %']]
    iso = IsolationForest(contamination=0.05, random_state=0)
    iso.fit(X)
    df['anomaly_score'] = -iso.decision_function(X)
    # emphasize sales by simple multiplication
    df['combined_score'] = df['anomaly_score'] * df['Продажа с ЗЦ сумма']
    return df

def display_table(df, title):
    st.subheader(f"{title}  (найдено {len(df)})")
    cols = ['Категория','Группа','Name_tov',
            'Списания %','Закрытие потребности %',
            'Продажа с ЗЦ сумма','anomaly_score','combined_score']
    styler = df[cols].style.format({
        'Списания %':'{:.1f}', 'Закрытие потребности %':'{:.1f}',
        'Продажа с ЗЦ сумма':'{:.0f}',
        'anomaly_score':'{:.3f}', 'combined_score':'{:.0f}'
    }).background_gradient(subset=['Списания %'], cmap='Reds') \
     .background_gradient(subset=['Закрытие потребности %'], cmap='Blues') \
     .background_gradient(subset=['combined_score'], cmap='Purples')
    st.dataframe(styler, use_container_width=True)

def main():
    st.title("Анализ аномалий: списания и закрытие потребности")

    uploaded = st.file_uploader("CSV со статистикой", type="csv")
    if not uploaded:
        st.stop()
    df = load_and_prepare(uploaded)
    df = compute_anomaly(df)

    st.sidebar.header("Настройки фильтрации")
    # sales range
    min_sale = float(df['Продажа с ЗЦ сумма'].min())
    max_sale = float(df['Продажа с ЗЦ сумма'].max())
    sale_range = st.sidebar.slider("Сумма продаж (руб.)", min_sale, max_sale, (min_sale, max_sale))
    # thresholds for % lists and fill
    low_waste = st.sidebar.slider("Низкие списания %", 0.0, 50.0, (0.5, 8.0))
    low_fill  = st.sidebar.slider("Низкие закрытие %", 0.0, 100.0, (10.0, 75.0))
    high_waste = st.sidebar.slider("Высокие списания %", 0.0, 200.0, 20.0)
    high_fill  = st.sidebar.slider("Высокие закрытие %", 0.0, 200.0, 80.0)

    # apply filters
    df = df[
        (df['Продажа с ЗЦ сумма'] >= sale_range[0]) &
        (df['Продажа с ЗЦ сумма'] <= sale_range[1])
    ]
    low_cond = df['Списания %'].between(low_waste[0], low_waste[1]) & \
               df['Закрытие потребности %'].between(low_fill[0], low_fill[1])
    high_cond = (df['Списания %'] >= high_waste) & (df['Закрытие потребности %'] >= high_fill)

    low_df = df[low_cond].sort_values('combined_score', ascending=False)
    high_df = df[high_cond].sort_values('combined_score', ascending=False)

    display_table(low_df, "Низкие списания +    низкое закрытие потребности")
    display_table(high_df, "Высокие списания +    высокое закрытие потребности")

    # download
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        low_df.to_excel(writer, sheet_name='Низкие', index=False)
        high_df.to_excel(writer, sheet_name='Высокие', index=False)
    buf.seek(0)
    st.download_button("Скачать результаты в Excel", buf,
                       "anomalies.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

if __name__=="__main__":
    main()
