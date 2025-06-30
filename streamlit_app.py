import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from sklearn.ensemble import IsolationForest
import plotly.express as px
from st_aggrid import AgGrid, GridOptionsBuilder

st.set_page_config(page_title="Аномалии: списания & закрытие", layout="wide")

@st.cache_data(show_spinner=False)
def load_and_prepare(uploaded):
    name = uploaded.name.lower()
    if name.endswith((".xls", ".xlsx")):
        df = pd.read_excel(uploaded, engine="openpyxl")
    else:
        df = pd.read_csv(uploaded)
    df.columns = df.columns.str.strip()

    # Переименование ключевых колонок
    col_map = {}
    if 'parent_group_name' in df.columns:
        col_map['parent_group_name'] = 'Категория'
    if 'group_name' in df.columns:
        col_map['group_name'] = 'Группа'
    fmt = next((c for c in df.columns if 'формат' in c.lower() or 'format' in c.lower()), None)
    if fmt:
        col_map[fmt] = 'Формат'
    skl = next((c for c in df.columns if 'name_sklad' in c.lower()), None)
    if skl:
        col_map[skl] = 'Склад'
    else:
        skl = next((c for c in df.columns if 'sklad' in c.lower()), None)
        if skl:
            col_map[skl] = 'Склад'
    df = df.rename(columns=col_map)

    required = ['Категория', 'Группа', 'Name_tov', 'Формат', 'Склад']
    for req in required:
        if req not in df.columns:
            raise KeyError(f"Missing column '{req}' in input file")

    # Выручка
    sale_col = next((c for c in df.columns if 'продажа' in c.lower() and 'сумма' in c.lower()), None)
    if not sale_col:
        raise KeyError("Missing sales sum column")
    df['Продажа с ЗЦ сумма'] = pd.to_numeric(df[sale_col], errors='coerce').fillna(0)

    # Закрытие потребности
    fill_col = next((c for c in df.columns if 'закрытие' in c.lower()), None)
    if not fill_col:
        raise KeyError("Missing fill percentage column")
    df['Закрытие потребности %'] = pd.to_numeric(
        df[fill_col].astype(str).str.replace(',', '.').str.rstrip('%'), errors='coerce') * 100

    # Списания
    zc2 = next((c for c in df.columns if c.lower().startswith('зц2')), None)
    srok = next((c for c in df.columns if c.lower() == 'срок'), None)
    kach = next((c for c in df.columns if c.lower() == 'качество'), None)
    if not all([zc2, srok, kach]):
        raise KeyError("Missing waste columns ZC2, срок, or качество")
    for col in (zc2, srok, kach):
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    df['Списания %'] = np.where(
        df['Продажа с ЗЦ сумма'] > 0,
        (df[zc2] + df[srok] + df[kach]) / df['Продажа с ЗЦ сумма'] * 100,
        0
    )

    df = df.dropna(subset=required + ['Списания %', 'Закрытие потребности %'])

        # Агрегация
    grp_cols = ['Категория', 'Группа', 'Name_tov', 'Формат', 'Склад']
    grp = df.groupby(grp_cols)
    total = grp['Продажа с ЗЦ сумма'].sum()

    # Безопасное вычисление средневзвешенных значений
    def weighted_avg(x, col):
        sales = x['Продажа с ЗЦ сумма']
        if sales.sum() > 0:
            return np.average(x[col], weights=sales)
        else:
            return x[col].mean()

    waste = grp.apply(lambda x: weighted_avg(x, 'Списания %'), include_groups=False)
    fill = grp.apply(lambda x: weighted_avg(x, 'Закрытие потребности %'), include_groups=False)

    agg = pd.concat([total, waste, fill], axis=1).reset_index()
    agg.columns = grp_cols + ['Продажа с ЗЦ сумма', 'Списания %', 'Закрытие потребности %']

    # Групповые метрики
    agg['group_mean_waste'] = agg.groupby('Группа')['Списания %'].transform('mean')
    # Средневзвешенное по группам с защитой от нулевого веса
    wmap = agg.groupby('Группа').apply(
        lambda g: weighted_avg(g, 'Списания %'), include_groups=False
    ).to_dict()
    agg['group_weighted_mean_waste'] = agg['Группа'].map(wmap)

    return agg


@st.cache_data(show_spinner=False)
def score_anomalies(df):
    df = df.copy()
    df['anomaly_score'] = 0.0
    for grp_name, sub in df.groupby('Группа'):
        if len(sub) < 2:
            continue
        X = sub[['Списания %', 'Закрытие потребности %']]
        iso = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1, n_estimators=30)
        iso.fit(X)
        df.loc[sub.index, 'anomaly_score'] = -iso.decision_function(X)
    df['anomaly_severity'] = df['anomaly_score'].abs()
    df['sales_share_in_group'] = df['Продажа с ЗЦ сумма'] / df.groupby('Группа')['Продажа с ЗЦ сумма'].transform('sum').fillna(1)
    df['combined_score'] = df['anomaly_severity'] * df['sales_share_in_group']
    return df


def display_anomaly_table(df, title):
    st.subheader(f"{title} (найдено {len(df)})")
    cols = ['Категория', 'Группа', 'Формат', 'Склад', 'Name_tov', 'Списания %',
            'group_mean_waste', 'group_weighted_mean_waste', 'Закрытие потребности %',
            'Продажа с ЗЦ сумма', 'anomaly_severity', 'combined_score']
    rename_map = {
        'group_mean_waste': 'Среднее в группе %',
        'group_weighted_mean_waste': 'Средневзв. в группе %',
        'anomaly_severity': 'Степень аномалии',
        'combined_score': 'Скор в группе'
    }
    styler = (
        df[cols]
        .rename(columns=rename_map)
        .style
        .format({
            'Списания %': '{:.1f}',
            'Среднее в группе %': '{:.1f}',
            'Средневзв. в группе %': '{:.1f}',
            'Закрытие потребности %': '{:.1f}',
            'Продажа с ЗЦ сумма': '{:.0f}',
            'Степень аномалии': '{:.3f}',
            'Скор в группе': '{:.3f}'
        })
        .background_gradient(subset=['Списания %'], cmap='Reds')
        .background_gradient(subset=['Закрытие потребности %'], cmap='Blues')
        .background_gradient(subset=['Скор в группе'], cmap='Purples')
    )
    st.dataframe(styler, use_container_width=True)


def main():
    st.title("Аномалии: списания и закрытие потребности")
    uploaded = st.file_uploader("Загрузите CSV/Excel", type=['csv', 'xls', 'xlsx'])
    if not uploaded:
        return

    full_df = score_anomalies(load_and_prepare(uploaded))
    st.write("**Первые 10 комбинаций (для отладки)**", 
             full_df[['Name_tov','Формат','Склад','Списания %','Закрытие потребности %']].head(10))
    df = full_df.copy()

    # Сайдбар: фильтрация и пресеты
    sb = st.sidebar
    sb.header("Фильтрация")
    sel_cats = sb.multiselect("Категории", sorted(df['Категория'].unique()), default=df['Категория'].unique())
    sel_grps = sb.multiselect("Группы", sorted(df[df['Категория'].isin(sel_cats)]['Группа'].unique()), default=None)
    sel_fmts = sb.multiselect("Форматы ТТ", sorted(df['Формат'].unique()), default=None)
    sel_whs = sb.multiselect("Склады", sorted(df['Склад'].unique()), default=None)
    if sel_cats:
        df = df[df['Категория'].isin(sel_cats)]
    if sel_grps:
        df = df[df['Группа'].isin(sel_grps)]
    if sel_fmts:
        df = df[df['Формат'].isin(sel_fmts)]
    if sel_whs:
        df = df[df['Склад'].isin(sel_whs)]
    if df.empty:
        st.warning("Нет данных после фильтров")
        return

    # Слайдеры выручки
    smin, smax = int(df['Продажа с ЗЦ сумма'].min()), int(df['Продажа с ЗЦ сумма'].max())
    sel_rng = sb.slider("Выручка (₽)", smin, smax, (smin, smax), step=1)
    df = df[df['Продажа с ЗЦ сумму'].between(sel_rng[0], sel_rng[1])]

    sb.markdown("---")
    sb.header("Чувствительность")
    preset = sb.radio("Пресет", ["Нет", "Слабая", "Средняя", "Высокая"], index=2)
    defs = {
        "Нет":     ((0.0,100.0), (0.0,100.0), 0.0,0.0),
        "Слабая":  ((0.5,15.0),  (5.0,85.0),  15.0,60.0),
        "Средняя": ((0.5,8.0),   (10.0,75.0), 20.0,80.0),
        "Высокая": ((0.5,5.0),   (20.0,60.0), 25.0,90.0)
    }
    lw_def, lf_def, hw_def, hf_def = defs[preset]
    low_min, low_max = sb.slider("Списания % (диапазон)", 0.0,100.0, lw_def, step=0.1)
    close_min, close_max = sb.slider("Закрытие % (диапазон)", 0.0,100.0, lf_def, step=0.1)
    hw_thr = sb.slider("Порог списания %", 0.0,200.0, hw_def, step=0.1)
    hf_thr = sb.slider("Порог закрытия %", 0.0,200.0, hf_def, step=0.1)

    low_df = df[(df['Списания %'].between(low_min, low_max)) &
                (df['Закрытие потребности %'].between(close_min, close_max))]
    high_df = df[(df['Списания %'] >= hw_thr) & (df['Закрытие потребности %'] >= hf_thr)]

    display_anomaly_table(low_df.sort_values('combined_score', ascending=False).head(100), "Низкие списания + низкое закрытие (топ-100)")
    display_anomaly_table(high_df.sort_values('combined_score', ascending=False).head(100), "Высокие списания + высокое закрытие (топ-100)")

    # График разброса
    df_plot = df.copy()
    df_plot['Статус'] = np.where(df_plot.index.isin(pd.concat([low_df, high_df]).index), 'Аномалия', 'Норма')
    fig = px.scatter(
        df_plot, x='Списания %', y='Закрытие потребности %',
        color='Статус', size='Продажа с ЗЦ сумма', opacity=0.6,
        hover_data=['Name_tov','Группа','Формат','Склад'],
        color_discrete_map={'Норма':'lightgray','Аномалия':'crimson'}
    )
    st.plotly_chart(fig, use_container_width=True)

    # Экспорт
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        low_df.to_excel(writer, sheet_name='Низкие', index=False)
        high_df.to_excel(writer, sheet_name='Высокие', index=False)
    buf.seek(0)
    st.download_button("Скачать Excel", buf, "anomalies.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # Иерархическое сравнение внутри main()
    st.header("Сравнение по форматам и складам (древовидно)")
    with st.expander("Показать/скрыть полное иерархическое сравнение"):
        comp = full_df.groupby(['Склад','Формат','Категория','Группа']).agg({
            'Списания %':'mean',
            'Закрытие потребности %':'mean',
            'Продажа с ЗЦ сумма':'sum'
        }).reset_index()
        comp = comp[['Склад','Формат','Категория','Группа','Списания %','Закрытие потребности %']]

        gb = GridOptionsBuilder.from_dataframe(comp)
        gb.configure_grid_options(
            treeData=True,
            animateRows=True,
            groupDefaultExpanded=0,
            getDataPath=['Склад','Формат','Категория','Группа']
        )
        gb.configure_default_column(enableRowGroup=True, rowGroup=True, hide=True)
                        # Настройка колонок с форматированием значений
        gb.configure_columns(
            ['Списания %','Закрытие потребность %'],
            type=['numericColumn'],
            aggFunc='mean',
            valueFormatter="params.value.toFixed(1) + '%'"
        )
        gridOptions = gb.build()
        AgGrid(
            comp,
            gridOptions=gridOptions,
            fit_columns_on_grid_load=True,
            height=500,
            enable_enterprise_modules=True,
            allow_unsafe_jscode=True
        )

if __name__ == "__main__":
    main()
