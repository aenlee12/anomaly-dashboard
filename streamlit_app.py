import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from sklearn.ensemble import IsolationForest
import plotly.express as px

st.set_page_config(page_title="Аномалии: списания & закрытие", layout="wide")

@st.cache_data
def load_and_prepare(uploaded):
    # 1) Читаем файл
    name = uploaded.name.lower()
    if name.endswith((".xls", ".xlsx")):
        df = pd.read_excel(uploaded, engine="openpyxl")
    else:
        df = pd.read_csv(uploaded)
    df.columns = df.columns.str.strip()

    # 2) Переименование колонок
    col_map = {}
    if 'parent_group_name' in df.columns:
        col_map['parent_group_name'] = 'Категория'
    if 'group_name' in df.columns:
        col_map['group_name'] = 'Группа'
    # Формат ТТ
    fmt = next((c for c in df.columns if 'формат' in c.lower() or 'format' in c.lower()), None)
    if fmt:
        col_map[fmt] = 'Формат'
    # Склад
    skl = next((c for c in df.columns if 'name_sklad' in c.lower()), None)
    if not skl:
        skl = next((c for c in df.columns if 'sklad' in c.lower()), None)
    if skl:
        col_map[skl] = 'Склад'
    df = df.rename(columns=col_map)

    # 3) Проверка обязательных полей
    for req in ['Категория', 'Группа', 'Name_tov', 'Формат', 'Склад']:
        if req not in df.columns:
            raise KeyError(f"В файле нет колонки «{req}»")

    # 4) Считаем продажи (по сумме)
    sale_col = next((c for c in df.columns 
                     if 'продажа' in c.lower() and 'сумма' in c.lower()), None)
    if sale_col is None:
        raise KeyError("Не найдена колонка с суммой продаж")
    df['Продажа с ЗЦ сумма'] = pd.to_numeric(df[sale_col], errors='coerce').fillna(0)

    # 5) Считаем % закрытия и переводим в проценты 0-100
    fill_col = next(c for c in df.columns if 'закрытие' in c.lower())
    df['Закрытие потребности %'] = (
        pd.to_numeric(
            df[fill_col].astype(str)
                  .str.replace(',', '.')
                  .str.rstrip('%'),
            errors='coerce'
        ) * 100
    )

    # 6) Считаем «Списания %» по формуле (ZC2 + срок + качество) / Продажа_с_ЗЦ_сумма * 100
    col_zc2  = next((c for c in df.columns if c.lower().startswith('зц2')), None)
    col_srok = next((c for c in df.columns if c.lower() == 'срок'), None)
    col_kach = next((c for c in df.columns if c.lower() == 'качество'), None)
    if not (col_zc2 and col_srok and col_kach):
        raise KeyError(f"Не найдены колонки для списания: ZC2={col_zc2}, срок={col_srok}, качество={col_kach}")
    df[col_zc2]  = pd.to_numeric(df[col_zc2],  errors='coerce').fillna(0)
    df[col_srok] = pd.to_numeric(df[col_srok], errors='coerce').fillna(0)
    df[col_kach] = pd.to_numeric(df[col_kach], errors='coerce').fillna(0)
    df['Списания %'] = np.where(
        df['Продажа с ЗЦ сумма'] > 0,
        (df[col_zc2] + df[col_srok] + df[col_kach]) / df['Продажа с ЗЦ сумма'] * 100,
        0
    )

    # 7) Убираем неинформативные
    df = df.dropna(subset=['Категория','Группа','Name_tov','Списания %','Закрытие потребности %'])

    # 8) Сохраняем метаданные формат/склад для склеивания после агрегации
    meta = df[['Категория','Группа','Name_tov','Формат','Склад']].drop_duplicates()

    # 9) Агрегируем по SKU
    def agg_group(g):
        tot   = g['Продажа с ЗЦ сумма'].sum()
        waste = np.average(g['Списания %'],            weights=g['Продажа с ЗЦ сумма']) if tot>0 else g['Списания %'].mean()
        fill  = np.average(g['Закрытие потребности %'], weights=g['Продажа с ЗЦ сумма']) if tot>0 else g['Закрытие потребности %'].mean()
        return pd.Series({
            'Списания %':             waste,
            'Закрытие потребности %': fill,
            'Продажа с ЗЦ сумма':     tot
        })
    agg_df = df.groupby(['Категория','Группа','Name_tov'], as_index=False).apply(agg_group)

    # 10) Внутригрупповые метрики
    agg_df['group_mean_waste'] = agg_df.groupby('Группа')['Списания %'].transform('mean')
    wmap = agg_df.groupby('Группа').apply(
        lambda g: np.average(g['Списания %'], weights=g['Продажа с ЗЦ сумма']) if g['Продажа с ЗЦ сумма'].sum()>0 else g['Списания %'].mean()
    ).to_dict()
    agg_df['group_weighted_mean_waste'] = agg_df['Группа'].map(wmap)

    # 11) Склеиваем формат и склад обратно
    return agg_df.merge(meta, on=['Категория','Группа','Name_tov'], how='left')


@st.cache_data
def score_anomalies(df):
    df = df.copy()
    df['anomaly_score'] = 0.0
    for grp, sub in df.groupby('Группа'):
        X = sub[['Списания %','Закрытие потребности %']]
        if len(sub) < 2:
            continue
        iso = IsolationForest(
            contamination=0.05,
            random_state=42,
            n_jobs=-1,
            n_estimators=50
        )
        iso.fit(X)
        df.loc[sub.index, 'anomaly_score'] = -iso.decision_function(X)
    df['anomaly_severity'] = df['anomaly_score'].abs()
    df['sales_share_in_group'] = (
        df['Продажа с ЗЦ сумма'] /
        df.groupby('Группа')['Продажа с ЗЦ сумма'].transform('sum')
    ).fillna(0)
    df['combined_score'] = df['anomaly_severity'] * df['sales_share_in_group']
    return df


def display_anomaly_table(df, title):
    st.subheader(f"{title} (найдено {len(df)})")
    cols = [
        'Категория','Группа','Формат','Склад','Name_tov',
        'Списания %','group_mean_waste','group_weighted_mean_waste',
        'Закрытие потребности %','Продажа с ЗЦ сумма',
        'anomaly_severity','combined_score'
    ]
    rename_map = {
        'group_mean_waste':           'Среднее в группе %',
        'group_weighted_mean_waste':  'Средневзв. в группе %',
        'anomaly_severity':           'Степень аномалии',
        'combined_score':             'Скор в группе'
    }
    styler = (
        df[cols]
          .rename(columns=rename_map)
          .style.format({
              'Списания %':              '{:.1f}',
              'Среднее в группе %':      '{:.1f}',
              'Средневзв. в группе %':   '{:.1f}',
              'Закрытие потребности %':  '{:.1f}',
              'Продажа с ЗЦ сумма':      '{:.0f}',
              'Степень аномалии':        '{:.3f}',
              'Скор в группе':           '{:.3f}'
          })
          .background_gradient(subset=['Списания %'],            cmap='Reds')
          .background_gradient(subset=['Закрытие потребности %'],cmap='Blues')
          .background_gradient(subset=['Скор в группе'],        cmap='Purples')
    )
    st.dataframe(styler, use_container_width=True)


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
    df = df[
        df['Категория'].isin(sel_cats) &
        df['Группа'].isin(sel_grps) &
        df['Формат'].isin(sel_fmts) &
        df['Склад'].isin(sel_whs)
    ]
    if df.empty:
        st.warning("Нет данных после фильтров")
        return

    # Выручка
    smin, smax = int(df['Продажа с ЗЦ сумма'].min()), int(df['Продажа с ЗЦ сумма'].max())
    sel_range = sb.slider("Выручка (₽)", smin, smax, (smin, smax), step=1)
    min_rev = sb.number_input("Мин. выручка (₽)", smin, smax, value=sel_range[0], step=1)
    max_rev = sb.number_input("Макс. выручка (₽)", smin, smax, value=sel_range[1], step=1)
    df = df[df['Продажа с ЗЦ сумма'].between(min_rev, max_rev)]

    # Пороги
    sb.header("Чувствительность")
    preset = sb.radio("Пресет", ["Нет","Слабая","Средняя","Высокая"])
    defs = {
        "Слабая":  ((0.5,15.0), (5.0,85.0), 15.0, 60.0),
        "Средняя": ((0.5,8.0),  (10.0,75.0), 20.0, 80.0),\``
