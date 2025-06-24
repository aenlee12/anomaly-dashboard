import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from sklearn.ensemble import IsolationForest
import plotly.express as px

st.set_page_config(page_title="Аномалии: списания & закрытие", layout="wide")

@st.cache_data
def load_and_prepare(uploaded):
    name = uploaded.name.lower()
    if name.endswith((".xls", ".xlsx")):
        df = pd.read_excel(uploaded, engine="openpyxl")
    else:
        df = pd.read_csv(uploaded)
    df.columns = df.columns.str.strip()

    # Переименование колонок
    if 'Категория' not in df.columns:
        if 'parent_group_name' in df.columns:
            df = df.rename(columns={'parent_group_name': 'Категория'})
        else:
            raise KeyError("Отсутствует колонка 'Категория' или 'parent_group_name'")
    if 'Группа' not in df.columns:
        if 'group_name' in df.columns:
            df = df.rename(columns={'group_name': 'Группа'})
        else:
            raise KeyError("Отсутствует колонка 'Группа' или 'group_name'")

    # Автопоиск процентных и суммовых полей
    waste_col = next((c for c in df.columns if 'срок' in c.lower() and 'качество' in c.lower()), None)
    fill_col = next((c for c in df.columns if 'закрытие' in c.lower()), None)
    sale_col = next((c for c in df.columns if 'продажа' in c.lower()), None)
    if not waste_col or not fill_col or not sale_col:
        raise KeyError(f"Нужные колонки не найдены: списания={waste_col}, закрытие={fill_col}, продажи={sale_col}")

    df['Списания %'] = pd.to_numeric(
        df[waste_col].astype(str).str.replace(',', '.').str.rstrip('%'), errors='coerce')
    df['Закрытие потребности %'] = pd.to_numeric(
        df[fill_col].astype(str).str.replace(',', '.').str.rstrip('%'), errors='coerce')
    df['Продажа с ЗЦ сумма'] = pd.to_numeric(df[sale_col], errors='coerce').fillna(0)

    df = df.dropna(subset=['Категория', 'Группа', 'Name_tov', 'Списания %', 'Закрытие потребности %'])

    def agg(g):
        tot = g['Продажа с ЗЦ сумма'].sum()
        waste = np.average(g['Списания %'], weights=g['Продажа с ЗЦ сумма']) if tot>0 else g['Списания %'].mean()
        fill = np.average(g['Закрытие потребности %'], weights=g['Продажа с ЗЦ сумма']) if tot>0 else g['Закрытие потребности %'].mean()
        return pd.Series({'Списания %': waste, 'Закрытие потребности %': fill, 'Продажа с ЗЦ сумма': tot})

    df = df.groupby(['Категория','Группа','Name_tov'], as_index=False).apply(agg)
    df['group_mean_waste'] = df.groupby('Группа')['Списания %'].transform('mean')
    weighted = df.groupby('Группа').apply(
        lambda g: np.average(g['Списания %'], weights=g['Продажа с ЗЦ сумма']) if g['Продажа с ЗЦ сумма'].sum()>0 else g['Списания %'].mean()
    ).to_dict()
    df['group_weighted_mean_waste'] = df['Группа'].map(weighted)
    return df

@st.cache_data
def score_anomalies(df):
    df = df.copy()
    df['anomaly_score'] = 0.0
    for grp, sub in df.groupby('Группа'):
        X = sub[['Списания %','Закрытие потребности %']]
        iso = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1, n_estimators=50)
        iso.fit(X)
        df.loc[sub.index,'anomaly_score'] = -iso.decision_function(X)
    df['anomaly_severity'] = df['anomaly_score'].abs()
    df['sales_share_in_group'] = df['Продажа с ЗЦ сумма'] / df.groupby('Группа')['Продажа с ЗЦ сумма'].transform('sum')
    df['combined_score'] = df['anomaly_severity'] * df['sales_share_in_group'].fillna(0)
    return df


def display_anomaly_table(df, title):
    st.subheader(f"{title} (найдено {len(df)})")
    cols = ['Категория','Группа','Name_tov','Списания %','group_mean_waste','group_weighted_mean_waste',
            'Закрытие потребности %','Продажа с ЗЦ сумма','anomaly_severity','combined_score']
    rename = {
        'group_mean_waste':'Среднее в группе %', 'group_weighted_mean_waste':'Средневзв. в группе %',
        'anomaly_severity':'Степень аномалии','combined_score':'Скор в группе'
    }
    styler = df[cols].rename(columns=rename).style.format({
        'Списания %':'{:.1f}','Среднее в группе %':'{:.1f}','Средневзв. в группе %':'{:.1f}',
        'Закрытие потребности %':'{:.1f}','Продажа с ЗЦ сумма':'{:.0f}','Степень аномалии':'{:.3f}','Скор в группе':'{:.3f}'
    }).background_gradient(subset=['Списания %'], cmap='Reds')\
      .background_gradient(subset=['Закрытие потребности %'], cmap='Blues')\
      .background_gradient(subset=['Скор в группе'], cmap='Purples')
    st.dataframe(styler, use_container_width=True)


def main():
    st.title("Анализ аномалий: списания и закрытие потребности")
    uploaded = st.file_uploader("Загрузите файл (CSV или Excel)")
    if not uploaded:
        return
    df = load_and_prepare(uploaded)
    df = score_anomalies(df)

    # фильтры категорий и групп
    cats = sorted(df['Категория'].unique())
    sel_cats = st.sidebar.multiselect("Категории", cats, default=cats)
    df = df[df['Категория'].isin(sel_cats)]
    grps = sorted(df['Группа'].unique())
    query = st.sidebar.text_input("Поиск групп", "")
    sel_grps = [g for g in grps if query.lower() in g.lower()]
    sel_grps = st.sidebar.multiselect("Группы", sel_grps, default=sel_grps)
    df = df[df['Группа'].isin(sel_grps)]
    if df.empty:
        st.warning("Нет данных после фильтрации — измените параметры")
        return

    # выручка
    min_sale = int(df['Продажа с ЗЦ сумма'].min())
    max_sale = int(df['Продажа с ЗЦ сумма'].max())
    sale_range = st.sidebar.slider("Выручка (₽)", min_sale, max_sale, (min_sale, max_sale), step=1)
    df = df[df['Продажа с ЗЦ сумма'].between(sale_range[0], sale_range[1])]

    # пресеты
    preset = st.sidebar.radio("Пресет чувствительности", ["Нет","Слабая","Средняя","Высокая"])
    if preset == "Нет":
        lw_def, lf_def, hw_def, hf_def = (0.0,100.0),(0.0,100.0),0.0,0.0
    elif preset == "Слабая":
        lw_def, lf_def, hw_def, hf_def = (0.5,15.0),(5.0,85.0),15.0,60.0
    elif preset == "Средняя":
        lw_def, lf_def, hw_def, hf_def = (0.5,8.0),(10.0,75.0),20.0,80.0
    else:
        lw_def, lf_def, hw_def, hf_def = (0.5,5.0),(20.0,60.0),25.0,90.0

    # слайдеры
    low = st.sidebar.slider("Списания % диапазон", 0.0, 100.0, lw_def, step=0.1)
    close = st.sidebar.slider("Закрытие % диапазон", 0.0, 100.0, lf_def, step=0.1)
    high_waste = st.sidebar.slider("Порог списания %", 0.0, 200.0, hw_def, step=0.1)
    high_close = st.sidebar.slider("Порог закрытия %", 0.0, 200.0, hf_def, step=0.1)

    # отбор
    low_df = df[df['Списания %'].between(*low) & df['Закрытие потребности %'].between(*close)].sort_values('combined_score', ascending=False)
    high_df = df[(df['Списания %']>=high_waste) & (df['Закрытие потребности %']>=high_close)].sort_values('combined_score', ascending=False)

    display_anomaly_table(low_df, "Низкие списания + низкое закрытие")
    display_anomaly_table(high_df, "Высокие списания + высокое закрытие")

    # график
    mask = df.index.isin(pd.concat([low_df, high_df]).index)
    df_plot = df.copy()
    df_plot['Статус'] = np.where(mask,'Аномалия','Норма')
    fig = px.scatter(df_plot, x='Списания %', y='Закрытие потребности %', color='Статус', size='Продажа с ЗЦ сумма', opacity=0.6, hover_data=['Name_tov','Группа'], color_discrete_map={'Норма':'lightgrey','Аномалия':'crimson'})
    st.plotly_chart(fig, use_container_width=True)

    # экспорт
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as w:
        low_df.to_excel(w, sheet_name='Низкие', index=False)
        high_df.to_excel(w, sheet_name='Высокие', index=False)
    buf.seek(0)
    st.download_button("Скачать в Excel", buf, "anomalies.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

if __name__ == "__main__":
    main()
