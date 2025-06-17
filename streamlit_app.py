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
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(X)
    raw = model.decision_function(X)
    df['anomaly_score'] = (-raw).clip(lower=0)
    df['anomaly_severity'] = df['anomaly_score']
    df['combined_score'] = df['anomaly_severity'] * df['Продажа с ЗЦ сумма']
    return df

@st.cache_data
def score_by_group(df):
    df = df.copy()
    df['group_anomaly_score'] = np.nan
    for grp, sub in df.groupby('Группа'):
        Xg = sub[['Списания %','Закрытие потребности %']]
        model = IsolationForest(contamination=0.05, random_state=42)
        model.fit(Xg)
        raw = model.decision_function(Xg)
        df.loc[sub.index, 'group_anomaly_score'] = (-raw).clip(lower=0)
    df['group_anomaly_severity'] = df['group_anomaly_score']
    df['group_combined_score'] = df['group_anomaly_severity'] * df['Продажа с ЗЦ сумма']
    return df

def display_table(df, cols, rename_map, title):
    st.subheader(f"{title}  (найдено {len(df)})")
    styled = df[cols].rename(columns=rename_map).style.format({
        'Списания %': '{:.1f}',
        'Закрытие потребности %': '{:.1f}',
        'Продажа с ЗЦ сумма': '{:.0f}',
        'Доля в группе': '{:.2%}',
        **{rename_map.get('Степень аномалии', ''): '{:.3f}'},
        **{rename_map.get('Скор (руб.)',''): '{:.0f}'},
        **{rename_map.get('Скор в группе',''): '{:.0f}'},
    })
    # Apply color gradients
    styled = (
        styled
        .background_gradient(subset=['Списания %'],            cmap='Reds')
        .background_gradient(subset=['Закрытие потребности %'],cmap='Blues')
    )
    if 'Скор (руб.)' in rename_map.values():
        styled = styled.background_gradient(subset=[rename_map['combined_score']], cmap='Purples')
    else:
        styled = styled.background_gradient(subset=[rename_map['group_combined_score']], cmap='Purples')
    st.dataframe(styled, use_container_width=True)

def main():
    st.title("Анализ аномалий: списания и закрытие потребности")
    uploaded = st.file_uploader("Загрузите CSV-файл", type="csv")
    if not uploaded:
        return

    df = load_and_prepare(uploaded)
    df = score_global(df)
    df = score_by_group(df)
    # relative sales share
    df['sales_share_in_group'] = (
        df['Продажа с ЗЦ сумма']
        / df.groupby('Группа')['Продажа с ЗЦ сумма'].transform('sum')
    ).fillna(0)

    # Sidebar presets & sliders
    sale_min, sale_max = float(df['Продажа с ЗЦ сумма'].min()), float(df['Продажа с ЗЦ сумма'].max())
    preset = st.sidebar.radio("Пресет чувствительности", ["Нет", "Слабая", "Средняя", "Высокая"])
    if preset == "Слабая":
        sale_def, lw_def, lf_def, hw_def, hf_def = (sale_min, sale_max), (0.5,15), (5,85), 15, 60
    elif preset == "Средняя":
        sale_def, lw_def, lf_def, hw_def, hf_def = (sale_min, sale_max), (0.5,8), (10,75), 20, 80
    elif preset == "Высокая":
        sale_def, lw_def, lf_def, hw_def, hf_def = (sale_min, sale_max), (0.5,5), (20,60), 25, 90
    else:
        sale_def, lw_def, lf_def, hw_def, hf_def = (sale_min, sale_max), (0.5,8), (10,75), 20, 80

    st.sidebar.header("Настройки фильтрации")
    sale_range = st.sidebar.slider("Сумма продаж (руб.)", sale_min, sale_max, sale_def)
    st.sidebar.divider()
    low_waste = st.sidebar.slider("Низкие списания %", 0.0, 100.0, lw_def)
    low_fill  = st.sidebar.slider("Низкое закрытие %", 0.0, 100.0, lf_def)
    st.sidebar.divider()
    high_waste = st.sidebar.slider("Высокие списания % порог", 0.0, 200.0, hw_def)
    high_fill  = st.sidebar.slider("Высокое закрытие % порог", 0.0, 200.0, hf_def)

    # Filter by sales
    df_filtered = df[
        df['Продажа с ЗЦ сумма'].between(sale_range[0], sale_range[1])
    ]

    # Global anomalies
    low_global = df_filtered[
        df_filtered['Списания %'].between(*low_waste) &
        df_filtered['Закрытие потребности %'].between(*low_fill)
    ].sort_values('combined_score', ascending=False)
    high_global = df_filtered[
        (df_filtered['Списания %'] >= high_waste) &
        (df_filtered['Закрытие потребности %'] >= high_fill)
    ].sort_values('combined_score', ascending=False)

    # Group anomalies
    low_group = df_filtered[
        df_filtered['Списания %'].between(*low_waste) &
        df_filtered['Закрытие потребности %'].between(*low_fill)
    ].sort_values('group_combined_score', ascending=False)
    high_group = df_filtered[
        (df_filtered['Списания %'] >= high_waste) &
        (df_filtered['Закрытие потребности %'] >= high_fill)
    ].sort_values('group_combined_score', ascending=False)

    # Display tables
    cols_global = ['Категория','Группа','Name_tov','Списания %','Закрытие потребности %',
                   'Продажа с ЗЦ сумма','sales_share_in_group','anomaly_severity','combined_score']
    rename_global = {
        'sales_share_in_group':'Доля в группе',
        'anomaly_severity':'Степень аномалии',
        'combined_score':'Скор (руб.)'
    }
    display_table(low_global, cols_global, rename_global, "Низкие аномалии (глобальный скор)")
    display_table(high_global, cols_global, rename_global, "Высокие аномалии (глобальный скор)")

    cols_group = ['Категория','Группа','Name_tov','Списания %','Закрытие потребности %',
                  'Продажа с ЗЦ сумма','sales_share_in_group','group_anomaly_severity','group_combined_score']
    rename_group = {
        'sales_share_in_group':'Доля в группе',
        'group_anomaly_severity':'Степень аномалии',
        'group_combined_score':'Скор в группе'
    }
    display_table(low_group, cols_group, rename_group, "Низкие аномалии (групповой скор)")
    display_table(high_group, cols_group, rename_group, "Высокие аномалии (групповой скор)")

    # Export
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        low_global.to_excel(writer, sheet_name='Low_Global', index=False)
        high_global.to_excel(writer, sheet_name='High_Global', index=False)
        low_group.to_excel(writer, sheet_name='Low_Group', index=False)
        high_group.to_excel(writer, sheet_name='High_Group', index=False)
    buf.seek(0)
    st.download_button("Скачать Excel", buf, "anomalies_full.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

if __name__ == "__main__":
    main()

