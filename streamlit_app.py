import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from sklearn.ensemble import IsolationForest
import plotly.express as px

st.set_page_config(page_title="Аномалии: списания & закрытие", layout="wide")

@st.cache_data
"
"def score_anomalies(df):
"
"    df = df.copy()
"
"    df['anomaly_score'] = 0.0
"
"    # Теперь считаем аномалии в разрезе Группа+Формат+Склад
"
"    for (grp, fmt, wh), sub in df.groupby(['Группа','Формат','Склад']):
"
"        X = sub[['Списания %','Закрытие потребности %']]
"
"        if len(sub) < 2:
"
"            # недостаточно точек для модели, оставляем нули
"
"            continue
"
"        iso = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1, n_estimators=50)
"
"        iso.fit(X)
"
"        df.loc[sub.index, 'anomaly_score'] = -iso.decision_function(X)
"
"    df['anomaly_severity'] = df['anomaly_score'].abs()
"
"    # Доля продаж внутри составной группы
"
"    df['sales_share_in_group'] = df['Продажа с ЗЦ сумма'] / df.groupby(['Группа','Формат','Склад'])['Продажа с ЗЦ сумма'].transform('sum')
"
"    df['combined_score'] = df['anomaly_severity'] * df['sales_share_in_group'].fillna(0)
"
"    return df


def display_anomaly_table(df, title):
    st.subheader(f"{title} (найдено {len(df)})")
    cols = ['Категория','Группа','Формат','Склад','Name_tov',
            'Списания %','group_mean_waste','group_weighted_mean_waste',
            'Закрытие потребности %','Продажа с ЗЦ сумма','anomaly_severity','combined_score']
    rename_map = {'group_mean_waste':'Среднее в группе %','group_weighted_mean_waste':'Средневзв. в группе %','anomaly_severity':'Степень аномалии','combined_score':'Скор в группе'}
    st.dataframe(
        df[cols].rename(columns=rename_map)
          .style.format({'Списания %':'{:.1f}','Среднее в группе %':'{:.1f}','Средневзв. в группе %':'{:.1f}','Закрытие потребности %':'{:.1f}','Продажа с ЗЦ сумма':'{:.0f}','Степень аномалии':'{:.3f}','Скор в группе':'{:.3f}'})
          .background_gradient(subset=['Списания %'],cmap='Reds')
          .background_gradient(subset=['Закрытие потребности %'],cmap='Blues')
          .background_gradient(subset=['Скор в группе'],cmap='Purples'),
        use_container_width=True
    )


def main():
    st.title("Анализ аномалий: списания и закрытие потребности")
    uploaded = st.file_uploader("Загрузите CSV/Excel", type=['csv','xls','xlsx'])
    if not uploaded:
        return
    df = load_and_prepare(uploaded)
    df = score_anomalies(df)

    sb = st.sidebar
    sb.header("Фильтрация")
    cats = sorted(df['Категория'].unique())
    sel_cats = sb.multiselect("Категории", cats, default=cats)
    grps = sorted(df.loc[df['Категория'].isin(sel_cats),'Группа'].unique())
    sel_grps = sb.multiselect("Группы", grps, default=grps)
    fmts = sorted(df['Формат'].unique())
    sel_fmts = sb.multiselect("Форматы ТТ", fmts, default=fmts)
    whs = sorted(df['Склад'].unique())
    sel_whs = sb.multiselect("Склады", whs, default=whs)
    df = df[df['Категория'].isin(sel_cats) & df['Группа'].isin(sel_grps) & df['Формат'].isin(sel_fmts) & df['Склад'].isin(sel_whs)]
    if df.empty:
        st.warning("Нет данных после фильтров")
        return

    # Выручка с вводом
    smin, smax = int(df['Продажа с ЗЦ сумма'].min()), int(df['Продажа с ЗЦ сумма'].max())
    sel_range = sb.slider("Выручка (₽)", smin, smax, (smin, smax), step=1)
    min_rev = sb.number_input("Мин. выручка (₽)", smin, smax, value=sel_range[0], step=1)
    max_rev = sb.number_input("Макс. выручка (₽)", smin, smax, value=sel_range[1], step=1)
    df = df[df['Продажа с ЗЦ сумма'].between(min_rev, max_rev)]

    # Чувствительность и пороги с ручным вводом
    sb.header("Чувствительность")
    preset = sb.radio("Пресет", ["Нет","Слабая","Средняя","Высокая"])
    defs = {"Слабая": ((0.5,15.0),(5.0,85.0),15.0,60.0),"Средняя":((0.5,8.0),(10.0,75.0),20.0,80.0),"Высокая":((0.5,5.0),(20.0,60.0),25.0,90.0),"Нет":((0.0,100.0),(0.0,100.0),0.0,0.0)}
    lw_def, lf_def, hw_def, hf_def = defs[preset]
    sb.subheader("Низкие списания + низкое закрытие")
    low_rng = sb.slider("Списания % (диапазон)", 0.0, 100.0, lw_def, step=0.1)
    lw_min = sb.number_input("Мин. списания %", 0.0, 100.0, value=low_rng[0], step=0.1)
    lw_max = sb.number_input("Макс. списания %", 0.0, 100.0, value=low_rng[1], step=0.1)
    close_rng = sb.slider("Закрытие % (диапазон)", 0.0, 100.0, lf_def, step=0.1)
    lf_min = sb.number_input("Мин. закрытие %", 0.0, 100.0, value=close_rng[0], step=0.1)
    lf_max = sb.number_input("Макс. закрытие %", 0.0, 100.0, value=close_rng[1], step=0.1)
    sb.subheader("Высокие списания + высокое закрытие")
    hw_thr = sb.number_input("Порог списания %", 0.0, 200.0, value=hw_def, step=0.1)
    hf_thr = sb.number_input("Порог закрытия %", 0.0, 200.0, value=hf_def, step=0.1)

        low_df = df[
        df['Списания %'].between(lw_min, lw_max) &
        df['Закрытие потребности %'].between(lf_min, lf_max)
    ]
    high_df = df[(df['Списания %']>=hw_thr) & (    df['Закрытие потребности %'] = pd.to_numeric(
        df[fill_col].astype(str).str.replace(',','.').str.rstrip('%'), errors='coerce'
    )
    df['Списания %'] = pd.to_numeric(
        df[waste_col].astype(str).str.replace(',','.').str.rstrip('%'), errors='coerce'
    )
    df['Продажа с ЗЦ сумма'] = pd.to_numeric(df[sale_col], errors='coerce').fillna(0)
