import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from sklearn.ensemble import IsolationForest
import plotly.express as px

st.set_page_config(page_title="Аномалии: списания & закрытие", layout="wide")

@st.cache_data
def load_and_prepare(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df['Списания %'] = pd.to_numeric(
        df['ЗЦ2_срок_качество_%'].str.replace(',', '.').str.rstrip('%'),
        errors='coerce'
    )
    df['Закрытие потребности %'] = pd.to_numeric(
        df['Закрытие потребности_%'].str.replace(',', '.').str.rstrip('%'),
        errors='coerce'
    )
    df['Продажа с ЗЦ сумма'] = pd.to_numeric(
        df['Продажа_с_ЗЦ_сумма'], errors='coerce'
    ).fillna(0)
    return df.dropna(subset=['Списания %','Закрытие потребности %'])

@st.cache_data
def score_global(df):
    X = df[['Списания %','Закрытие потребности %']]
    iso = IsolationForest(contamination=0.05, random_state=42)
    iso.fit(X)
    raw = iso.decision_function(X)
    df['anomaly_score']    = -raw
    df['anomaly_severity'] = df['anomaly_score'].abs()
    df['combined_score']   = df['anomaly_severity'] * df['Продажа с ЗЦ сумма']
    return df

@st.cache_data
def score_by_group(df):
    df = df.copy()
    df['group_anomaly_score'] = 0.0
    for grp, sub in df.groupby('Группа'):
        Xg = sub[['Списания %','Закрытие потребности %']]
        iso = IsolationForest(contamination=0.05, random_state=42)
        iso.fit(Xg)
        raw = iso.decision_function(Xg)
        df.loc[sub.index, 'group_anomaly_score'] = -raw
    df['group_anomaly_severity'] = df['group_anomaly_score'].abs()
    df['group_combined_score']   = df['group_anomaly_severity'] * df['Продажа с ЗЦ сумма']
    return df

def display_table(df, cols, rename_map, title):
    st.subheader(f"{title}  (найдено {len(df)})")
    styled = (
        df[cols]
        .rename(columns=rename_map)
        .style.format({
            'Списания %':          '{:.1f}',
            'Закрытие потребности %': '{:.1f}',
            'Продажа с ЗЦ сумма':  '{:.0f}',
            **({'Доля в группе':'{:.2%}'} if 'Доля в группе' in rename_map.values() else {}),
            **({'Степень аномалии':'{:.3f}'} if 'Степень аномалии' in rename_map.values() else {}),
            **({'Скор (руб.)':'{:.0f}'} if 'Скор (руб.)' in rename_map.values() else {}),
            **({'Скор в группе':'{:.0f}'} if 'Скор в группе' in rename_map.values() else {}),
        })
        .background_gradient(subset=['Списания %'],            cmap='Reds')
        .background_gradient(subset=['Закрытие потребности %'],cmap='Blues')
    )
    if 'Скор (руб.)' in rename_map.values():
        styled = styled.background_gradient(subset=['Скор (руб.)'], cmap='Purples')
    else:
        styled = styled.background_gradient(subset=['Скор в группе'], cmap='Purples')
    st.dataframe(styled, use_container_width=True)

def main():
    st.title("Анализ аномалий: списания и закрытие потребности")
    uploaded = st.file_uploader("Загрузите CSV-файл", type="csv")
    if not uploaded:
        return

    # 1. Загрузка и скоринг
    df = load_and_prepare(uploaded)
    df = score_global(df)
    df = score_by_group(df)

    # 2. Доля продаж в группе
    df['sales_share_in_group'] = (
        df['Продажа с ЗЦ сумма']
        / df.groupby('Группа')['Продажа с ЗЦ сумма'].transform('sum')
    ).fillna(0)

    # 3. Пресеты и слайдеры
    sale_min, sale_max = df['Продажа с ЗЦ сумма'].min(), df['Продажа с ЗЦ сумма'].max()
    preset = st.sidebar.radio("Пресет чувствительности", ["Нет", "Слабая", "Средняя", "Высокая"])
    if preset == "Слабая":
        sale_def = (sale_min, sale_max)
        low_waste_def, low_fill_def = (0.5, 15.0), (5.0, 85.0)
        high_waste_def, high_fill_def = 15.0, 60.0
    elif preset == "Средняя":
        sale_def = (sale_min, sale_max)
        low_waste_def, low_fill_def = (0.5, 8.0), (10.0, 75.0)
        high_waste_def, high_fill_def = 20.0, 80.0
    elif preset == "Высокая":
        sale_def = (sale_min, sale_max)
        low_waste_def, low_fill_def = (0.5, 5.0), (20.0, 60.0)
        high_waste_def, high_fill_def = 25.0, 90.0
    else:
        sale_def = (sale_min, sale_max)
        low_waste_def, low_fill_def = (0.5, 8.0), (10.0, 75.0)
        high_waste_def, high_fill_def = 20.0, 80.0

    st.sidebar.header("Настройки фильтрации")
    sale_range = st.sidebar.slider("Сумма продаж (руб.)", sale_min, sale_max, sale_def, key="sale_range")
    st.sidebar.divider()
    low_waste = st.sidebar.slider("Низкие списания %", 0.0, 100.0, low_waste_def, key="low_waste")
    low_fill  = st.sidebar.slider("Низкое закрытие %", 0.0, 100.0, low_fill_def, key="low_fill")
    st.sidebar.divider()
    high_waste = st.sidebar.slider("Высокие списания % порог", 0.0, 200.0, high_waste_def, key="high_waste")
    high_fill  = st.sidebar.slider("Высокое закрытие % порог", 0.0, 200.0, high_fill_def, key="high_fill")

    # 4. Фильтрация
    df_f = df[df['Продажа с ЗЦ сумма'].between(*sale_range)]

    # 5. Глобальные аномалии
    low_gl = df_f[
        df_f['Списания %'].between(*low_waste) &
        df_f['Закрытие потребности %'].between(*low_fill)
    ].sort_values('combined_score', ascending=False)
    high_gl = df_f[
        (df_f['Списания %'] >= high_waste) &
        (df_f['Закрытие потребности %'] >= high_fill)
    ].sort_values('combined_score', ascending=False)

    # 6. Групповые аномалии
    low_gr = df_f[
        df_f['Списания %'].between(*low_waste) &
        df_f['Закрытие потребности %'].between(*low_fill)
    ].sort_values('group_combined_score', ascending=False)
    high_gr = df_f[
        (df_f['Списания %'] >= high_waste) &
        (df_f['Закрытие потребности %'] >= high_fill)
    ].sort_values('group_combined_score', ascending=False)

    # 7. Вывод таблиц
    cols_gl = [
        'Категория','Группа','Name_tov','Списания %','Закрытие потребности %',
        'Продажа с ЗЦ сумма','sales_share_in_group','anomaly_severity','combined_score'
    ]
    rename_gl = {
        'sales_share_in_group':'Доля в группе',
        'anomaly_severity':'Степень аномалии',
        'combined_score':'Скор (руб.)'
    }
    display_table(low_gl,  cols_gl, rename_gl, "Низкие аномалии (глобальный скор)")
    display_table(high_gl, cols_gl, rename_gl, "Высокие аномалии (глобальный скор)")

    cols_gr = [
        'Категория','Группа','Name_tov','Списания %','Закрытие потребности %',
        'Продажа с ЗЦ сумма','sales_share_in_group','group_anomaly_severity','group_combined_score'
    ]
    rename_gr = {
        'sales_share_in_group':'Доля в группе',
        'group_anomaly_severity':'Степень аномалии',
        'group_combined_score':'Скор в группе'
    }
    display_table(low_gr,  cols_gr, rename_gr, "Низкие аномалии (групповой скор)")
    display_table(high_gr, cols_gr, rename_gr, "Высокие аномалии (групповой скор)")

    # 8. Экспорт
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        low_gl .to_excel(writer, sheet_name='Low_Global',  index=False)
        high_gl.to_excel(writer, sheet_name='High_Global', index=False)
        low_gr .to_excel(writer, sheet_name='Low_Group',   index=False)
        high_gr.to_excel(writer, sheet_name='High_Group',  index=False)
    buf.seek(0)
    st.download_button(
        "Скачать Excel",
        buf,
        "anomalies_full.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

if __name__ == "__main__":
    main()
