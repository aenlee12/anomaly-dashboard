import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from sklearn.ensemble import IsolationForest
import plotly.express as px

st.set_page_config(page_title="Аномалии: списания & закрытие", layout="wide")

@st.cache_data
def load_and_prepare(uploaded):
    # 1) Читаем CSV или Excel
    name = uploaded.name.lower()
    if name.endswith((".xls", ".xlsx")):
        df = pd.read_excel(uploaded, engine="openpyxl")
    else:
        df = pd.read_csv(uploaded)
    df.columns = df.columns.str.strip()  # убираем лишние пробелы

    # 1.1) Переименовываем английские названия колонок, если нужно
    #    — если в файле уже есть русские "Категория"/"Группа", они сохранятся
    if 'Категория' not in df.columns:
        if 'parent_group_name' in df.columns:
            df = df.rename(columns={'parent_group_name': 'Категория'})
        else:
            raise KeyError("Не найдена колонка 'Категория' или 'parent_group_name'")
    if 'Группа' not in df.columns:
        if 'group_name' in df.columns:
            df = df.rename(columns={'group_name': 'Группа'})
        else:
            raise KeyError("Не найдена колонка 'Группа' или 'group_name'")

    # 2) Автоопределяем остальные нужные колонки
    waste_col = next((c for c in df.columns
                      if "срок" in c.lower() and "качество" in c.lower()), None)
    fill_col  = next((c for c in df.columns
                      if "закрытие" in c.lower()), None)
    sale_col  = next((c for c in df.columns
                      if "продажа" in c.lower()), None)
    if not (waste_col and fill_col and sale_col):
        raise KeyError(
            f"Не найдены требуемые колонки:\n"
            f"  списания ({waste_col}),\n"
            f"  закрытие ({fill_col}),\n"
            f"  продажи ({sale_col})"
        )

    # 3) Парсим проценты и суммы
    df['Списания %'] = pd.to_numeric(
        df[waste_col].astype(str).str.replace(',', '.').str.rstrip('%'),
        errors='coerce'
    )
    df['Закрытие потребности %'] = pd.to_numeric(
        df[fill_col].astype(str).str.replace(',', '.').str.rstrip('%'),
        errors='coerce'
    )
    df['Продажа с ЗЦ сумма'] = pd.to_numeric(
        df[sale_col], errors='coerce'
    ).fillna(0)

    # 4) Очищаем строки без ключевых полей
    df = df.dropna(subset=[
        'Категория', 'Группа', 'Name_tov',
        'Списания %', 'Закрытие потребности %'
    ])

    # 5) Агрегируем по SKU (категория–группа–товар)
    def agg_group(g):
        tot = g['Продажа с ЗЦ сумма'].sum()
        waste = (np.average(g['Списания %'], weights=g['Продажа с ЗЦ сумма'])
                 if tot > 0 else g['Списания %'].mean())
        fill  = (np.average(g['Закрытие потребности %'], weights=g['Продажа с ЗЦ сумма'])
                 if tot > 0 else g['Закрытие потребности %'].mean())
        return pd.Series({
            'Списания %':             waste,
            'Закрытие потребности %': fill,
            'Продажа с ЗЦ сумма':     tot
        })
    df = df.groupby(['Категория', 'Группа', 'Name_tov'], as_index=False).apply(agg_group)

    # 6) Внутригрупповые метрики
    df['group_mean_waste'] = df.groupby('Группа')['Списания %'].transform('mean')
    weighted = df.groupby('Группа').apply(
        lambda g: (np.average(g['Списания %'], weights=g['Продажа с ЗЦ сумма'])
                   if g['Продажа с ЗЦ сумма'].sum() > 0 else g['Списания %'].mean())
    ).to_dict()
    df['group_weighted_mean_waste'] = df['Группа'].map(weighted)

    return df

@st.cache_data
def score_anomalies(df):
    df = df.copy()
    df['anomaly_score'] = 0.0
    for grp, sub in df.groupby('Группа'):
        X = sub[['Списания %', 'Закрытие потребности %']]
        iso = IsolationForest(
            contamination=0.05, random_state=42,
            n_jobs=-1, n_estimators=50
        )
        iso.fit(X)
        raw = iso.decision_function(X)
        df.loc[sub.index, 'anomaly_score'] = -raw
    df['anomaly_severity'] = df['anomaly_score'].abs()
    df['sales_share_in_group'] = (
        df['Продажа с ЗЦ сумма'] /
        df.groupby('Группа')['Продажа с ЗЦ сумма'].transform('sum')
    ).fillna(0)
    df['combined_score'] = df['anomaly_severity'] * df['sales_share_in_group']
    return df

def display_anomaly_table(df, title):
    st.subheader(f"{title}  (найдено {len(df)})")
    cols = [
        'Категория','Группа','Name_tov',
        'Списания %','group_mean_waste','group_weighted_mean_waste',
        'Закрытие потребности %','Продажа с ЗЦ сумма',
        'anomaly_severity','combined_score'
    ]
    rename_map = {
        'group_mean_waste':          'Среднее в группе %',
        'group_weighted_mean_waste': 'Средневзв. в группе %',
        'anomaly_severity':          'Степень аномалии',
        'combined_score':            'Скор в группе'
    }
    styler = (
        df[cols]
          .rename(columns=rename_map)
          .style.format({
              'Списания %':             '{:.1f}',
              'Среднее в группе %':     '{:.1f}',
              'Средневзв. в группе %':  '{:.1f}',
              'Закрытие потребности %': '{:.1f}',
              'Продажа с ЗЦ сумма':     '{:.0f}',
              'Степень аномалии':       '{:.3f}',
              'Скор в группе':          '{:.3f}'
          })
          .background_gradient(subset=['Списания %'],            cmap='Reds')
          .background_gradient(subset=['Закрытие потребности %'],cmap='Blues')
          .background_gradient(subset=['Скор в группе'],        cmap='Purples')
    )
    st.dataframe(styler, use_container_width=True)

def main():
    st.title("Анализ аномалий: списания и закрытие потребности")
    uploaded = st.file_uploader("Загрузите файл (CSV или Excel)")
    if not uploaded:
        return

    df = load_and_prepare(uploaded)
    df = score_anomalies(df)

    # sidebar-фильтры (категории, группы, выручка, пресеты и т.д.)
    st.sidebar.header("Фильтрация по категориям")
    cats = sorted(df['Категория'].unique())
    sel_cats = st.sidebar.multiselect("Категории", cats, default=cats)
    df = df[df['Категория'].isin(sel_cats)]

    st.sidebar.header("Фильтрация по группам")
    grps = sorted(df['Группа'].unique())
    query = st.sidebar.text_input("Поиск групп", "")
    grps_f = [g for g in grps if query.lower() in g.lower()]
    sel_grps = st.sidebar.multiselect("Группы", grps_f, default=grps_f)
    df = df[df['Группа'].isin(sel_grps)]

    # … далее ваша логика пресетов, слайдеров и построения графиков без изменений …

    # пример скачивания
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as w:
        low_df.to_excel(w, sheet_name='Низкие', index=False)
        high_df.to_excel(w, sheet_name='Высокие', index=False)
    buf.seek(0)
    st.download_button(
        "Скачать в Excel", buf, "anomalies.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

if __name__ == "__main__":
    main()
