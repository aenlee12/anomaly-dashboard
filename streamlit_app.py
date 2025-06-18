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
        df['ЗЦ2_срок_качество_%']
          .str.replace(',', '.').str.rstrip('%'),
        errors='coerce'
    )
    df['Закрытие потребности %'] = pd.to_numeric(
        df['Закрытие потребности_%']
          .str.replace(',', '.').str.rstrip('%'),
        errors='coerce'
    )
    df['Продажа с ЗЦ сумма'] = pd.to_numeric(
        df['Продажа_с_ЗЦ_сумма'], errors='coerce'
    ).fillna(0)
    df = df.dropna(subset=['Списания %','Закрытие потребности %','Группа','Name_tov'])
    # Агрегация по периодам: объединяем возможные дубли по разным датам
    def agg_group(g):
        total_sales = g['Продажа с ЗЦ сумма'].sum()
        waste = (np.average(g['Списания %'], weights=g['Продажа с ЗЦ сумма'])
                 if total_sales>0 else g['Списания %'].mean())
        fill  = (np.average(g['Закрытие потребности %'], weights=g['Продажа с ЗЦ сумма'])
                 if total_sales>0 else g['Закрытие потребности %'].mean())
        return pd.Series({
            'Списания %': waste,
            'Закрытие потребности %': fill,
            'Продажа с ЗЦ сумма': total_sales
        })
    df = df.groupby(['Категория','Группа','Name_tov'], as_index=False).apply(agg_group)
    return df

@st.cache_data
def score_anomalies(df):
    df = df.copy()
    df['anomaly_score'] = 0.0
    # Считаем аномалии внутри каждой подгруппы
    for grp, sub in df.groupby('Группа'):
        X = sub[['Списания %','Закрытие потребности %']]
        iso = IsolationForest(contamination=0.05, random_state=42)
        iso.fit(X)
        raw = iso.decision_function(X)
        df.loc[sub.index, 'anomaly_score'] = -raw
    df['anomaly_severity'] = df['anomaly_score'].abs()
    # Доля выручки внутри группы
    df['sales_share_in_group'] = (
        df['Продажа с ЗЦ сумма'] /
        df.groupby('Группа')['Продажа с ЗЦ сумма'].transform('sum')
    ).fillna(0)
    # Итоговый скор в рамках группы
    df['combined_score'] = df['anomaly_severity'] * df['sales_share_in_group']
    return df

def display_anomaly_table(df, title):
    st.subheader(f"{title}  (найдено {len(df)})")
    cols = [
        'Категория','Группа','Name_tov',
        'Списания %','Закрытие потребности %',
        'Продажа с ЗЦ сумма','anomaly_severity','combined_score'
    ]
    rename_map = {
        'anomaly_severity': 'Степень аномалии',
        'combined_score':   'Скор в группе'
    }
    styler = (
        df[cols]
        .rename(columns=rename_map)
        .style.format({
            'Списания %':             '{:.1f}',
            'Закрытие потребности %': '{:.1f}',
            'Продажа с ЗЦ сумма':     '{:.0f}',
            'Степень аномалии':       '{:.3f}',
            'Скор в группе':          '{:.3f}'
        })
        .background_gradient(subset=['Списания %'],            cmap='Reds')
        .background_gradient(subset=['Закрытие потребности %'],cmap='Blues')
        .background_gradient(subset=['Скор в группе'],       cmap='Purples')
    )
    st.dataframe(styler, use_container_width=True)

def main():
    st.title("Анализ аномалий: списания и закрытие потребности")
    uploaded = st.file_uploader("Загрузите CSV-файл с периодами", type="csv")
    if not uploaded:
        return

    df = load_and_prepare(uploaded)
    df = score_anomalies(df)

    # Фильтры: ползунки + ручной ввод
    sale_min, sale_max = float(df['Продажа с ЗЦ сумма'].min()), float(df['Продажа с ЗЦ сумма'].max())
    st.sidebar.header("Фильтры по выручке")
    sale_range = st.sidebar.slider("Выручка (руб.)", sale_min, sale_max, (sale_min, sale_max))
    sale_min_in = st.sidebar.number_input("Мин. выручка", sale_min, sale_max, sale_range[0])
    sale_max_in = st.sidebar.number_input("Макс. выручка", sale_min, sale_max, sale_range[1])

    st.sidebar.header("Низкие аномалии")
    low_waste = st.sidebar.slider("Списания % [мин, макс]", 0.0, 100.0, (0.5, 8.0))
    low_waste_min = st.sidebar.number_input("Мин. списания %", 0.0, 100.0, low_waste[0])
    low_waste_max = st.sidebar.number_input("Макс. списания %", 0.0, 100.0, low_waste[1])
    low_fill  = st.sidebar.slider("Закрытие % [мин, макс]", 0.0, 100.0, (10.0, 75.0))
    low_fill_min  = st.sidebar.number_input("Мин. закрытие %", 0.0, 100.0, low_fill[0])
    low_fill_max  = st.sidebar.number_input("Макс. закрытие %", 0.0, 100.0, low_fill[1])

    st.sidebar.header("Высокие аномалии")
    high_waste = st.sidebar.slider("Порог списания %", 0.0, 200.0, 20.0)
    high_waste_thr = st.sidebar.number_input("Порог списания вручную", 0.0, 200.0, high_waste)
    high_fill  = st.sidebar.slider("Порог закрытия %", 0.0, 200.0, 80.0)
    high_fill_thr  = st.sidebar.number_input("Порог закрытия вручную", 0.0, 200.0, high_fill)

    # Применяем фильтры
    df = df[
        (df['Продажа с ЗЦ сумма'] >= sale_min_in) &
        (df['Продажа с ЗЦ сумма'] <= sale_max_in)
    ]
    low_df = df[
        df['Списания %'].between(low_waste_min, low_waste_max) &
        df['Закрытие потребности %'].between(low_fill_min, low_fill_max)
    ].sort_values('combined_score', ascending=False)
    high_df = df[
        (df['Списания %'] >= high_waste_thr) &
        (df['Закрытие потребности %'] >= high_fill_thr)
    ].sort_values('combined_score', ascending=False)

    # Вывод
    display_anomaly_table(low_df,  "Низкие аномалии (низкие списания + закрытие)")
    display_anomaly_table(high_df, "Высокие аномалии (высокие списания + закрытие)")

    # Сравнительный график
    mask = df.index.isin(pd.concat([low_df, high_df]).index)
    df['Статус'] = np.where(mask, 'Аномалия', 'Норма')
    fig = px.scatter(
        df, x='Списания %', y='Закрытие потребности %',
        color='Статус', size='Продажа с ЗЦ сумма', opacity=0.6,
        hover_data=['Name_tov','Группа'],
        color_discrete_map={'Норма':'lightgrey','Аномалия':'crimson'}
    )
    st.plotly_chart(fig, use_container_width=True)

    # Экспорт
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        low_df.to_excel(writer, sheet_name='Low', index=False)
        high_df.to_excel(writer, sheet_name='High', index=False)
    buf.seek(0)
    st.download_button(
        "Скачать Excel", buf, "anomalies.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

if __name__ == "__main__":
    main()

