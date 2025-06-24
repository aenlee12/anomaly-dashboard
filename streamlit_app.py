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
    df.columns = df.columns.str.strip()
    
    # 2) Автоопределяем колонки
    # списания: ищем «срок» и «качество»
    waste_col = next((c for c in df.columns
                      if "срок" in c.lower() and "качество" in c.lower()), None)
    # закрытие: ищем «закрытие»
    fill_col  = next((c for c in df.columns
                      if "закрытие" in c.lower()), None)
    # продажи: ищем «продажа»
    sale_col  = next((c for c in df.columns
                      if "продажа" in c.lower()), None)
    if not (waste_col and fill_col and sale_col):
        raise KeyError(f"Не найдены требуемые колонки: списания({waste_col}), закрытие({fill_col}), продажи({sale_col})")
    
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
    
    # 4) Очищаем
    df = df.dropna(subset=['Категория','Группа','Name_tov','Списания %','Закрытие потребности %'])
    
    # 5) Агрегируем по SKU (категория–группа–товар)
    def agg_group(g):
        tot = g['Продажа с ЗЦ сумма'].sum()
        waste = np.average(g['Списания %'], weights=g['Продажа с ЗЦ сумма']) if tot>0 else g['Списания %'].mean()
        fill  = np.average(g['Закрытие потребности %'], weights=g['Продажа с ЗЦ сумма']) if tot>0 else g['Закрытие потребности %'].mean()
        return pd.Series({
            'Списания %':              waste,
            'Закрытие потребности %':  fill,
            'Продажа с ЗЦ сумма':      tot
        })
    df = df.groupby(['Категория','Группа','Name_tov'], as_index=False).apply(agg_group)
    
    # 6) Внутригрупповые метрики
    df['group_mean_waste'] = df.groupby('Группа')['Списания %'].transform('mean')
    weighted = df.groupby('Группа').apply(
        lambda g: np.average(g['Списания %'], weights=g['Продажа с ЗЦ сумма'])
                  if g['Продажа с ЗЦ сумма'].sum()>0 else g['Списания %'].mean()
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
        raw = iso.decision_function(X)
        df.loc[sub.index, 'anomaly_score'] = -raw
    df['anomaly_severity']   = df['anomaly_score'].abs()
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
        'group_mean_waste':            'Среднее в группе %',
        'group_weighted_mean_waste':   'Средневзв. в группе %',
        'anomaly_severity':            'Степень аномалии',
        'combined_score':              'Скор в группе'
    }
    styler = (
        df[cols]
          .rename(columns=rename_map)
          .style.format({
              'Списания %':                     '{:.1f}',
              'Среднее в группе %':             '{:.1f}',
              'Средневзв. в группе %':          '{:.1f}',
              'Закрытие потребности %':         '{:.1f}',
              'Продажа с ЗЦ сумма':             '{:.0f}',
              'Степень аномалии':               '{:.3f}',
              'Скор в группе':                  '{:.3f}'
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

    # фильтрация
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

    # пресеты
    sale_min, sale_max = df['Продажа с ЗЦ сумма'].min(), df['Продажа с ЗЦ сумма'].max()
    preset = st.sidebar.radio("Пресет чувствительности", ["Нет","Слабая","Средняя","Высокая"])
    if preset=="Слабая":
        sale_def, lw_def, lf_def, hw_def, hf_def = ((sale_min,sale_max),(0.5,15),(5,85),15,60)
    elif preset=="Средняя":
        sale_def, lw_def, lf_def, hw_def, hf_def = ((sale_min,sale_max),(0.5,8),(10,75),20,80)
    else:
        sale_def, lw_def, lf_def, hw_def, hf_def = ((sale_min,sale_max),(0.5,5),(20,60),25,90)

    # фильтры
    st.sidebar.header("Выручка")
    sr = st.sidebar.slider("Выручка (₽)", sale_min, sale_max, sale_def, step=1)
    min_sale = st.sidebar.number_input("Мин. выручка", sale_min, sale_max, value=sr[0])
    max_sale = st.sidebar.number_input("Макс. выручка", sale_min, sale_max, value=sr[1])

    st.sidebar.header("Низкие списания + закрытие")
    lw = st.sidebar.slider("Списания % диапазон",0.0,100.0,lw_def,step=0.1)
    min_lw = st.sidebar.number_input("Мин. списания %",0.0,100.0,value=lw[0])
    max_lw = st.sidebar.number_input("Макс. списания %",0.0,100.0,value=lw[1])
    lf = st.sidebar.slider("Закрытие % диапазон",0.0,100.0,lf_def,step=0.1)
    min_lf = st.sidebar.number_input("Мин. закрытие %",0.0,100.0,value=lf[0])
    max_lf = st.sidebar.number_input("Макс. закрытие %",0.0,100.0,value=lf[1])

    st.sidebar.header("Высокие списания + закрытие")
    hw = st.sidebar.slider("Порог списания %",0.0,200.0,hw_def,step=0.1)
    thr_hw = st.sidebar.number_input("Порог списания % вручную",0.0,200.0,value=hw)
    hf = st.sidebar.slider("Порог закрытия %",0.0,200.0,hf_def,step=0.1)
    thr_hf= st.sidebar.number_input("Порог закрытия % вручную",0.0,200.0,value=hf)

    # применяем
    df = df[(df['Продажа с ЗЦ сумма']>=min_sale)&(df['Продажа с ЗЦ сумма']<=max_sale)]
    low_df = df[(df['Списания %'].between(min_lw,max_lw))&
                (df['Закрытие потребности %'].between(min_lf,max_lf))]\
             .sort_values('combined_score',ascending=False)
    high_df= df[(df['Списания %']>=thr_hw)&(df['Закрытие потребности %']>=thr_hf)]\
             .sort_values('combined_score',ascending=False)

    display_anomaly_table(low_df,  "Низкие списания + низкое закрытие")
    display_anomaly_table(high_df, "Высокие списания + высокое закрытие")

    mask = df.index.isin(pd.concat([low_df,high_df]).index)
    df_plot = df.copy(); df_plot['Статус']=np.where(mask,'Аномалия','Норма')
    fig = px.scatter(df_plot,x='Списания %',y='Закрытие потребности %',
                     color='Статус',size='Продажа с ЗЦ сумма',opacity=0.6,
                     hover_data=['Name_tov','Группа'],
                     color_discrete_map={'Норма':'lightgrey','Аномалия':'crimson'})
    st.plotly_chart(fig,use_container_width=True)

    buf = BytesIO()
    with pd.ExcelWriter(buf,engine='openpyxl') as w:
        low_df.to_excel(w,sheet_name='Низкие',index=False)
        high_df.to_excel(w,sheet_name='Высокие',index=False)
    buf.seek(0)
    st.download_button("Скачать в Excel",buf,"anomalies.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

if __name__=="__main__":
    main()
