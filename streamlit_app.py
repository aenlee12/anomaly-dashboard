import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from sklearn.ensemble import IsolationForest
import plotly.express as px

st.set_page_config(page_title="Анализ аномалий", layout="wide")

@st.cache_data
def load_and_prepare(uploaded):
    # Чтение файла
    fname = uploaded.name.lower()
    if fname.endswith((".xls", ".xlsx")):
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
    fmt = next((c for c in df.columns if 'form' in c.lower()), None)
    if fmt:
        col_map[fmt] = 'Формат'
    skl = next((c for c in df.columns if 'name_sklad' in c.lower()), None)
    if not skl:
        skl = next((c for c in df.columns if 'sklad' in c.lower()), None)
    if skl:
        col_map[skl] = 'Склад'
    df = df.rename(columns=col_map)

    # Проверка
    for req in ['Категория','Группа','Name_tov','Формат','Склад']:
        if req not in df.columns:
            raise KeyError(f"Нет колонки '{req}' в файле")

    # Парсинг
    waste_col = next(c for c in df.columns if 'срок' in c.lower() and 'качество' in c.lower())
    fill_col = next(c for c in df.columns if 'закрытие' in c.lower())
    sale_col = next(c for c in df.columns if 'продажа' in c.lower())
    df['Списания %'] = pd.to_numeric(df[waste_col].astype(str).str.replace(',','.').str.rstrip('%'), errors='coerce')
    df['Закрытие потребности %'] = pd.to_numeric(df[fill_col].astype(str).str.replace(',','.').str.rstrip('%'), errors='coerce')
    df['Продажа с ЗЦ сумма'] = pd.to_numeric(df[sale_col], errors='coerce').fillna(0)

    df = df.dropna(subset=['Категория','Группа','Name_tov','Списания %','Закрытие потребности %'])

    # Сохраняем формат и склад
    meta = df[['Категория','Группа','Name_tov','Формат','Склад']].drop_duplicates()

    # Агрегация
    def agg(g):
        tot = g['Продажа с ЗЦ сумма'].sum()
        waste = np.average(g['Списания %'], weights=g['Продажа с ЗЦ сумма']) if tot>0 else g['Списания %'].mean()
        fill = np.average(g['Закрытие потребности %'], weights=g['Продажа с ЗЦ сумма']) if tot>0 else g['Закрытие потребности %'].mean()
        return pd.Series({'Списания %': waste, 'Закрытие потребности %': fill, 'Продажа с ЗЦ сумма': tot})
    agg_df = df.groupby(['Категория','Группа','Name_tov'], as_index=False).apply(agg)

    # Внутригрупповые метрики
    agg_df['group_mean_waste'] = agg_df.groupby('Группа')['Списания %'].transform('mean')
    wmap = agg_df.groupby('Группа').apply(lambda g: np.average(g['Списания %'], weights=g['Продажа с ЗЦ сумма']) if g['Продажа с ЗЦ сумма'].sum()>0 else g['Списания %'].mean()).to_dict()
    agg_df['group_weighted_mean_waste'] = agg_df['Группа'].map(wmap)

    # Объединение
    result = agg_df.merge(meta, on=['Категория','Группа','Name_tov'], how='left')
    return result

@st.cache_data
def score_anomalies(df):
    df = df.copy()
    df['anomaly_score'] = 0.0
    for grp, sub in df.groupby('Группа'):
        X = sub[['Списания %','Закрытие потребности %']]
        if len(sub) < 2:
            continue
        iso = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1, n_estimators=50)
        iso.fit(X)
        df.loc[sub.index, 'anomaly_score'] = -iso.decision_function(X)
    df['anomaly_severity'] = df['anomaly_score'].abs()
    df['sales_share_in_group'] = df['Продажа с ЗЦ сумма'] / df.groupby('Группа')['Продажа с ЗЦ сумма'].transform('sum')
    df['combined_score'] = df['anomaly_severity'] * df['sales_share_in_group'].fillna(0)
    return df

def display_anomaly_table(df, title):
    st.subheader(f"{title} (найдено {len(df)})")
    cols = ['Категория','Группа','Формат','Склад','Name_tov','Списания %','group_mean_waste','group_weighted_mean_waste','Закрытие потребности %','Продажа с ЗЦ сумма','anomaly_severity','combined_score']
    rename_map = {'group_mean_waste':'Среднее в группе %','group_weighted_mean_waste':'Средневзв. в группе %','anomaly_severity':'Степень аномалии','combined_score':'Скор в группе'}
    styled = df[cols].rename(columns=rename_map).style.format({'Списания %':'{:.1f}','Среднее в группе %':'{:.1f}','Средневзв. в группе %':'{:.1f}','Закрытие потребности %':'{:.1f}','Продажа с ЗЦ сумма':'{:.0f}','Степень аномалии':'{:.3f}','Скор в группе':'{:.3f}'})
    styled = styled.background_gradient(subset=['Списания %'], cmap='Reds').background_gradient(subset=['Закрытие потребности %'], cmap='Blues').background_gradient(subset=['Скор в группе'], cmap='Purples')
    st.dataframe(styled, use_container_width=True)

def main():
    st.title("Анализ аномалий: списания и закрытие потребности")
    uploaded = st.file_uploader("Загрузите CSV/Excel", type=['csv','xls','xlsx'])
    if not uploaded:
        return
    df = load_and_prepare(uploaded)
    df = score_anomalies(df)

    # Sidebar фильтры
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
    df = df[df['Категория'].isin(sel_cats)&df['Группа'].isin(sel_grps)&df['Формат'].isin(sel_fmts)&df['Склад'].isin(sel_whs)]
    if df.empty:
        st.warning("Нет данных после фильтров")
        return

    # Выручка
    smin, smax = int(df['Продажа с ЗЦ сумма'].min()), int(df['Продажа с ЗЦ сумма'].max())
    sel_rev = sb.slider("Выручка (₽)", smin, smax, (smin, smax), step=1)
    df = df[df['Продажа с ЗЦ сумма'].between(sel_rev[0], sel_rev[1])]

    # Чувствительность
    sb.header("Чувствительность")
    preset = sb.radio("Пресет", ["Нет","Слабая","Средняя","Высокая"])
    defs = {"Слабая":((0.5,15.0),(5.0,85.0),15.0,60.0),"Средняя":((0.5,8.0),(10.0,75.0),20.0,80.0),"Высокая":((0.5,5.0),(20.0,60.0),25.0,90.0),"Нет":((0.0,100.0),(0.0,100.0),0.0,0.0)}
    lw_def, lf_def, hw_def, hf_def = defs[preset]
    sb.subheader("Низкие списания + низкое закрытие")
    lw = sb.slider("Списания %",0.0,100.0,lw_def,step=0.1)
    lf = sb.slider("Закрытие %",0.0,100.0,lf_def,step=0.1)
    sb.subheader("Высокие списания + высокое закрытие")
    hw = sb.number_input("Порог списания %",0.0,200.0,value=hw_def,step=0.1)
    hf = sb.number_input("Порог закрытия %",0.0,200.0,value=hf_def,step=0.1)

    low_df = df[df['Списания %'].between(*lw)&df['Закрытие потребности %'].between(*lf)]
    high_df = df[(df['Списания %']>=hw)&(df['Закрытие потребности %']>=hf)]

    display_anomaly_table(low_df.sort_values('combined_score',ascending=False),"Низкие списания + низкое закрытие")
    display_anomaly_table(high_df.sort_values('combined_score',ascending=False),"Высокие списания + высокое закрытие")

    # График
    mask = df.index.isin(pd.concat([low_df,high_df]).index)
    df_plot = df.copy(); df_plot['Статус']=np.where(mask,'Аномалия','Норма')
    fig = px.scatter(df_plot,x='Списания %',y='Закрытие потребности %',color='Статус',size='Продажа с ЗЦ сумма',opacity=0.6,hover_data=['Name_tov','Группа','Формат','Склад'],color_discrete_map={'Норма':'lightgray','Аномалия':'crimson'})
    st.plotly_chart(fig,use_container_width=True)

    # Экспорт
    buf=BytesIO()
    with pd.ExcelWriter(buf,engine='openpyxl') as writer:
        low_df.to_excel(writer,sheet_name='Низкие',index=False)
        high_df.to_excel(writer,sheet_name='Высокие',index=False)
    buf.seek(0)
    st.download_button("Скачать Excel",buf,"anomalies.xlsx","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

if __name__=="__main__": main()
