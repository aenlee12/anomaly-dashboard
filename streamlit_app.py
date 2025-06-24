import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from sklearn.ensemble import IsolationForest
import plotly.express as px

st.set_page_config(page_title="Аномалии: списания & закрытие", layout="wide")

@st.cache_data
def load_and_prepare(uploaded):
    # Загрузка файла
    name = uploaded.name.lower()
    if name.endswith((".xls", ".xlsx")):
        df = pd.read_excel(uploaded, engine="openpyxl")
    else:
        df = pd.read_csv(uploaded)
    df.columns = df.columns.str.strip()

        # Переименование ключевых колонок (рус/англ)
    # Категория и Группа
    col_map = {}
    if 'parent_group_name' in df.columns:
        col_map['parent_group_name'] = 'Категория'
    if 'group_name' in df.columns:
        col_map['group_name'] = 'Группа'
    df = df.rename(columns=col_map)

    # Автовыбор колонки Формат
    fmt_col = next((c for c in df.columns if 'формат' in c.lower()), None)
    if fmt_col:
        df = df.rename(columns={fmt_col: 'Формат'})
    # Автовыбор колонки Склад
    wh_col = next((c for c in df.columns if 'склад' in c.lower()), None)
    if wh_col:
        df = df.rename(columns={wh_col: 'Склад'})

    # Проверяем наличие обязательных колонок
    for req in ['Категория','Группа','Name_tov','Формат','Склад']:
        if req not in df.columns:
            raise KeyError(f"Не найдена колонка '{req}'")

    # Поиск полей процентов и суммы
    for req in ['Категория','Группа','Name_tov','Формат','Склад']:
        if req not in df.columns:
            raise KeyError(f"Не найдена колонка '{req}'")

    # Поиск полей процентов и суммы
    waste_col = next(c for c in df.columns if 'срок' in c.lower() and 'качество' in c.lower())
    fill_col  = next(c for c in df.columns if 'закрытие' in c.lower())
    sale_col  = next(c for c in df.columns if 'продажа' in c.lower())

    df['Списания %'] = pd.to_numeric(
        df[waste_col].astype(str).str.replace(',','.').str.rstrip('%'), errors='coerce')
    df['Закрытие потребности %'] = pd.to_numeric(
        df[fill_col].astype(str).str.replace(',','.').str.rstrip('%'), errors='coerce')
    df['Продажа с ЗЦ сумма'] = pd.to_numeric(df[sale_col], errors='coerce').fillna(0)

    # Убираем пустые строки
    df = df.dropna(subset=['Категория','Группа','Name_tov','Списания %','Закрытие потребности %'])

    # Агрегация по SKU
    def agg(g):
        tot = g['Продажа с ЗЦ сумма'].sum()
        waste = np.average(g['Списания %'], weights=g['Продажа с ЗЦ сумма']) if tot>0 else g['Списания %'].mean()
        fill  = np.average(g['Закрытие потребности %'], weights=g['Продажа с ЗЦ сумма']) if tot>0 else g['Закрытие потребности %'].mean()
        return pd.Series({'Списания %': waste,
                          'Закрытие потребности %': fill,
                          'Продажа с ЗЦ сумма': tot})
    df = df.groupby(['Категория','Группа','Name_tov','Формат','Склад'], as_index=False).apply(agg)

    # Внутригрупповые метрики
    df['group_mean_waste'] = df.groupby('Группа')['Списания %'].transform('mean')
    weights = df.groupby('Группа').apply(
        lambda g: np.average(g['Списания %'], weights=g['Продажа с ЗЦ сумма']) if g['Продажа с ЗЦ сумма'].sum()>0 else g['Списания %'].mean()
    ).to_dict()
    df['group_weighted_mean_waste'] = df['Группа'].map(weights)

    return df

@st.cache_data
def score_anomalies(df):
    df = df.copy()
    df['anomaly_score'] = 0.0
    for grp, sub in df.groupby('Группа'):
        X = sub[['Списания %','Закрытие потребности %']]
        iso = IsolationForest(contamination=0.05, random_state=42,
                              n_jobs=-1, n_estimators=50)
        iso.fit(X)
        df.loc[sub.index, 'anomaly_score'] = -iso.decision_function(X)
    df['anomaly_severity'] = df['anomaly_score'].abs()
    df['sales_share_in_group'] = (
        df['Продажа с ЗЦ сумма']/
        df.groupby('Группа')['Продажа с ЗЦ сумма'].transform('sum')
    ).fillna(0)
    df['combined_score'] = df['anomaly_severity'] * df['sales_share_in_group']
    return df


def display_anomaly_table(df, title):
    st.subheader(f"{title} (найдено {len(df)})")
    cols = ['Категория','Группа','Формат','Склад','Name_tov',
            'Списания %','group_mean_waste','group_weighted_mean_waste',
            'Закрытие потребности %','Продажа с ЗЦ сумма',
            'anomaly_severity','combined_score']
    rename_map = {
        'group_mean_waste':'Среднее в группе %',
        'group_weighted_mean_waste':'Средневзв. в группе %',
        'anomaly_severity':'Степень аномалии',
        'combined_score':'Скор в группе'
    }
    styler = (
        df[cols]
          .rename(columns=rename_map)
          .style.format({
            'Списания %':'{:.1f}',
            'group_mean_waste':'{:.1f}',
            'group_weighted_mean_waste':'{:.1f}',
            'Закрытие потребности %':'{:.1f}',
            'Продажа с ЗЦ сумма':'{:.0f}',
            'anomaly_severity':'{:.3f}',
            'combined_score':'{:.3f}'
          })
          .background_gradient(subset=['Списания %'], cmap='Reds')
          .background_gradient(subset=['Закрытие потребности %'], cmap='Blues')
          .background_gradient(subset=['combined_score'], cmap='Purples')
    )
    st.dataframe(styler, use_container_width=True)


def main():
    st.title("Анализ аномалий: списания и закрытие потребности")
    uploaded = st.file_uploader("Загрузите файл (CSV или Excel)", type=['csv','xls','xlsx'])
    if not uploaded:
        return

    df = load_and_prepare(uploaded)
    df = score_anomalies(df)

    # Фильтрация: категории, группы, формат, склад
    st.sidebar.header("Фильтрация")
    cats = sorted(df['Категория'].unique())
    sel_cats = st.sidebar.multiselect("Категории", cats, default=cats)
    grps = sorted(df.loc[df['Категория'].isin(sel_cats),'Группа'].unique())
    sel_grps = st.sidebar.multiselect("Группы", grps, default=grps)
    fmts = sorted(df['Формат'].unique())
    sel_fmts = st.sidebar.multiselect("Форматы ТТ", fmts, default=fmts)
    whs = sorted(df['Склад'].unique())
    sel_whs = st.sidebar.multiselect("Склады", whs, default=whs)

    df = df[
        df['Категория'].isin(sel_cats) &
        df['Группа'].isin(sel_grps) &
        df['Формат'].isin(sel_fmts) &
        df['Склад'].isin(sel_whs)
    ]
    if df.empty:
        st.warning("Нет данных после фильтрации — измените параметры")
        return

    # Фильтрация по выручке и чувствительности
    sale_min, sale_max = int(df['Продажа с ЗЦ сумма'].min()), int(df['Продажа с ЗЦ сумма'].max())
    sale_range = st.sidebar.slider("Выручка (₽)", sale_min, sale_max, (sale_min, sale_max), step=1)
    df = df[df['Продажа с ЗЦ сумма'].between(*sale_range)]

    preset = st.sidebar.radio("Пресет чувствительности", ["Нет","Слабая","Средняя","Высокая"])
    if preset == "Слабая":
        lw_def, lf_def, hw_def, hf_def = (0.5,15),(5,85),15,60
    elif preset == "Средняя":
        lw_def, lf_def, hw_def, hf_def = (0.5,8),(10,75),20,80
    elif preset == "Высокая":
        lw_def, lf_def, hw_def, hf_def = (0.5,5),(20,60),25,90
    else:
        lw_def, lf_def, hw_def, hf_def = (0.0,100),(0.0,100),0,0

    low = st.sidebar.slider("Списания % диапазон", 0.0, 100.0, lw_def, step=0.1)
    close = st.sidebar.slider("Закрытие % диапазон", 0.0, 100.0, lf_def, step=0.1)
    high_waste = st.sidebar.slider("Порог списание %", 0.0, 200.0, hw_def, step=0.1)
    high_close = st.sidebar.slider("Порог закрытие %", 0.0, 200.0, hf_def, step=0.1)

    low_df = df[df['Списания %'].between(*low) & df['Закрытие потребности %'].between(*close)]
    high_df = df[(df['Списания %']>=high_waste) & (df['Закрытие потребности %']>=high_close)]

    display_anomaly_table(low_df.sort_values('combined_score', ascending=False), "Низкие списания + низкое закрытие")
    display_anomaly_table(high_df.sort_values('combined_score', ascending=False), "Высокие списания + высокое закрытие")

    # График
    mask = df.index.isin(pd.concat([low_df, high_df]).index)
    df_plot = df.copy()
    df_plot['Статус'] = np.where(mask,'Аномалия','Норма')
    fig = px.scatter(df_plot, x='Списания %', y='Закрытие потребности %',
                     color='Статус', size='Продажа с ЗЦ сумма', opacity=0.6,
                     hover_data=['Name_tov','Группа','Формат','Склад'],
                     color_discrete_map={'Норма':'lightgrey','Аномалия':'crimson'})
    st.plotly_chart(fig, use_container_width=True)

    # Экспорт
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        low_df.to_excel(writer, sheet_name='Низкие', index=False)
        high_df.to_excel(writer, sheet_name='Высокие', index=False)
    buf.seek(0)
    st.download_button("Скачать Excel", buf, "anomalies.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

if __name__ == "__main__":
    main()
