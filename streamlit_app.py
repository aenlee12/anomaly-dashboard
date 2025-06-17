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

    # group averages of waste
    df['group_mean_waste'] = df.groupby('Группа')['Списания %'].transform('mean')
    # weighted mean waste per group
    g = df.groupby('Группа').apply(
        lambda x: np.average(x['Списания %'], weights=x['Продажа с ЗЦ сумма'])
    )
    df['group_weighted_mean_waste'] = df['Группа'].map(g.to_dict())
    return df.dropna(subset=['Списания %','Закрытие потребности %'])

@st.cache_data
def score_anomalies(df):
    X = df[['Списания %','Закрытие потребности %']]
    iso = IsolationForest(contamination=0.05, random_state=42)
    iso.fit(X)
    raw = iso.decision_function(X)
    df['anomaly_score']    = -raw
    df['anomaly_severity'] = df['anomaly_score'].abs()
    df['combined_score']   = df['anomaly_severity'] * df['Продажа с ЗЦ сумма']
    df['sales_share_in_group'] = (
        df['Продажа с ЗЦ сумма'] /
        df.groupby('Группа')['Продажа с ЗЦ сумма'].transform('sum')
    ).fillna(0)
    df['combined_score_rel'] = df['anomaly_severity'] * df['sales_share_in_group']
    return df

def display_anomaly_table(df, title):
    st.subheader(f"{title}  (найдено {len(df)})")
    cols = [
        'Категория','Группа','Name_tov',
        'Списания %','Закрытие потребности %',
        'Продажа с ЗЦ сумма','sales_share_in_group',
        'group_mean_waste','group_weighted_mean_waste',
        'anomaly_severity','combined_score','combined_score_rel'
    ]
    rename_map = {
        'sales_share_in_group':            'Доля в группе',
        'group_mean_waste':                'Среднее списание в группе %',
        'group_weighted_mean_waste':       'Средневзв. списание в группе %',
        'anomaly_severity':                'Степень аномалии',
        'combined_score':                  'Скор (руб.)',
        'combined_score_rel':              'Скор (отн.)'
    }
    styler = df[cols].rename(columns=rename_map).style.format({
        'Списания %':                       '{:.1f}',
        'Закрытие потребности %':           '{:.1f}',
        'Продажа с ЗЦ сумма':               '{:.0f}',
        'Доля в группе':                    '{:.2%}',
        'Среднее списание в группе %':      '{:.1f}',
        'Средневзв. списание в группе %':   '{:.1f}',
        'Степень аномалии':                '{:.3f}',
        'Скор (руб.)':                      '{:.0f}',
        'Скор (отн.)':                      '{:.2%}'
    })
    styler = (
        styler
        .background_gradient(subset=['Списания %'], cmap='Reds')
        .background_gradient(subset=['Закрытие потребности %'], cmap='Blues')
        .background_gradient(subset=['Скор (руб.)'], cmap='Purples')
    )
    st.dataframe(styler, use_container_width=True)

def main():
    st.title("Анализ аномалий: списания и закрытие потребности")
    uploaded = st.file_uploader("Загрузите CSV", type="csv")
    if not uploaded:
        return

    df = load_and_prepare(uploaded)
    df = score_anomalies(df)

    sale_min, sale_max = df['Продажа с ЗЦ сумма'].min(), df['Продажа с ЗЦ сумма'].max()
    preset = st.sidebar.radio("Пресет",["Нет","Слабая","Средняя","Высокая"])
    if preset=="Слабая":
        sale_def=(sale_min,sale_max);low_waste_def=(0.5,15);low_fill_def=(5,85);high_waste_def=15;high_fill_def=60
    elif preset=="Средняя":
        sale_def=(sale_min,sale_max);low_waste_def=(0.5,8);low_fill_def=(10,75);high_waste_def=20;high_fill_def=80
    elif preset=="Высокая":
        sale_def=(sale_min,sale_max);low_waste_def=(0.5,5);low_fill_def=(20,60);high_waste_def=25;high_fill_def=90
    else:
        sale_def=(sale_min,sale_max);low_waste_def=(0.5,8);low_fill_def=(10,75);high_waste_def=20;high_fill_def=80

    st.sidebar.header("Фильтры")
    sale_range = st.sidebar.slider("Выручка", sale_min, sale_max, sale_def)
    st.sidebar.divider()
    low_waste = st.sidebar.slider("Низкие списания %", 0, 100, low_waste_def)
    low_fill  = st.sidebar.slider("Низ. закрытие %", 0, 100, low_fill_def)
    st.sidebar.divider()
    high_waste= st.sidebar.slider("Высокие списания %", 0, 200, high_waste_def)
    high_fill = st.sidebar.slider("Выс. закрытие %", 0, 200, high_fill_def)

    df = df[df['Продажа с ЗЦ сумма'].between(*sale_range)]
    low_df = df[df['Списания %'].between(*low_waste)&df['Закрытие потребности %'].between(*low_fill)].sort_values('combined_score',ascending=False)
    high_df= df[(df['Списания %']>=high_waste)&(df['Закрытие потребности %']>=high_fill)].sort_values('combined_score',ascending=False)

    display_anomaly_table(low_df, "Низкие аномалии")
    display_anomaly_table(high_df,"Высокие аномалии")

    buf=BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as w:
        low_df.to_excel(w, sheet_name='Low',index=False)
        high_df.to_excel(w, sheet_name='High',index=False)
    buf.seek(0)
    st.download_button("Скачать Excel",buf, "results.xlsx","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

if __name__=="__main__":
    main()

