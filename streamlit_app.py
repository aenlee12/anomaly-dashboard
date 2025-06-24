import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from sklearn.ensemble import IsolationForest
import plotly.express as px

st.set_page_config(page_title="Аномалии: списания & закрытие", layout="wide")

@st.cache_data
def load_and_prepare(uploaded):
    # Загрузка и нормализация колонок
    name = uploaded.name.lower()
    if name.endswith((".xls", ".xlsx")):
        df = pd.read_excel(uploaded, engine="openpyxl")
    else:
        df = pd.read_csv(uploaded)
    df.columns = df.columns.str.strip()

        # Автоматическое переименование колонок
    rename_map = {}
    # Категория и Группа
    for eng, rus in [('parent_group_name','Категория'), ('group_name','Группа')]:
        if eng in df.columns:
            rename_map[eng] = rus
    # Формат (рус/лат)
    fmt_col = next((c for c in df.columns if 'формат' in c.lower() or 'format' in c.lower()), None)
    if fmt_col:
        rename_map[fmt_col] = 'Формат'
    # Склад (рус/лат)
    skl_col = next((c for c in df.columns if 'name_sklad' in c.lower()), None)
    if not skl_col:
        skl_col = next((c for c in df.columns if 'sklad' in c.lower()), None)
    if skl_col:
        rename_map[skl_col] = 'Склад'
    df = df.rename(columns=rename_map)

    # Проверка
    required = ['Категория','Группа','Name_tov','Формат','Склад']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Отсутствуют колонки: {missing}")

    # Проценты и суммы
    waste_col = next(c for c in df.columns if 'срок' in c.lower() and 'качество' in c.lower())
    fill_col = next(c for c in df.columns if 'закрытие' in c.lower())
    sale_col = next(c for c in df.columns if 'продажа' in c.lower())

    df['Списания %'] = pd.to_numeric(df[waste_col].astype(str).str.replace(',','.').str.rstrip('%'), errors='coerce')
    df['Закрытие потребности %'] = pd.to_numeric(df[fill_col].astype(str).str.replace(',','.').str.rstrip('%'), errors='coerce')
    df['Продажа с ЗЦ сумма'] = pd.to_numeric(df[sale_col], errors='coerce').fillna(0)

    df = df.dropna(subset=['Категория','Группа','Name_tov','Списания %','Закрытие потребности %'])

    # Агрегация
    def agg(g):
        tot = g['Продажа с ЗЦ сумма'].sum()
        waste = np.average(g['Списания %'], weights=g['Продажа с ЗЦ сумма']) if tot>0 else g['Списания %'].mean()
        fill = np.average(g['Закрытие потребности %'], weights=g['Продажа с ЗЦ сумма']) if tot>0 else g['Закрытие потребности %'].mean()
        return pd.Series({'Списания %':waste,'Закрытие потребности %':fill,'Продажа с ЗЦ сумма':tot})
    df = df.groupby(['Категория','Группа','Name_tov','Формат','Склад'], as_index=False).apply(agg)

    df['group_mean_waste'] = df.groupby('Группа')['Списания %'].transform('mean')
    w_map = df.groupby('Группа').apply(lambda x: np.average(x['Списания %'], weights=x['Продажа с ЗЦ сумма']) if x['Продажа с ЗЦ сумма'].sum()>0 else x['Списания %'].mean()).to_dict()
    df['group_weighted_mean_waste'] = df['Группа'].map(w_map)

    return df

@st.cache_data
def score_anomalies(df):
    df = df.copy()
    df['anomaly_score']=0
    for g,sub in df.groupby('Группа'):
        X=sub[['Списания %','Закрытие потребности %']]
        iso=IsolationForest(contamination=0.05,random_state=42,n_estimators=50,n_jobs=-1)
        iso.fit(X)
        df.loc[sub.index,'anomaly_score']=-iso.decision_function(X)
    df['anomaly_severity']=df['anomaly_score'].abs()
    df['sales_share_in_group']=df['Продажа с ЗЦ сумма']/df.groupby('Группа')['Продажа с ЗЦ сумма'].transform('sum')
    df['combined_score']=df['anomaly_severity']*df['sales_share_in_group'].fillna(0)
    return df

def display_anomaly_table(df,title):
    st.subheader(f"{title} (найдено {len(df)})")
    cols=['Категория','Группа','Формат','Склад','Name_tov','Списания %','group_mean_waste','group_weighted_mean_waste','Закрытие потребности %','Продажа с ЗЦ сумма','anomaly_severity','combined_score']
    rename={'group_mean_waste':'Среднее в группе %','group_weighted_mean_waste':'Средневзв. в группе %','anomaly_severity':'Степень аномалии','combined_score':'Скор в группе'}
    st.dataframe(
        df[cols].rename(columns=rename)
          .style.format({'Списания %':'{:.1f}','Среднее в группе %':'{:.1f}','Средневзв. в группе %':'{:.1f}','Закрытие потребности %':'{:.1f}','Продажа с ЗЦ сумма':'{:.0f}','Степень аномалии':'{:.3f}','Скор в группе':'{:.3f}'})
          .background_gradient(subset=['Списания %'],cmap='Reds')
          .background_gradient(subset=['Закрытие потребности %'],cmap='Blues')
          .background_gradient(subset=['Скор в группе'],cmap='Purples')
        ,use_container_width=True)


def main():
    st.title("Анализ аномалий: списания и закрытие потребности")
    uploaded=st.file_uploader("Загрузите CSV/Excel",type=['csv','xls','xlsx'])
    if not uploaded: return
    df=load_and_prepare(uploaded)
    df=score_anomalies(df)

    # Фильтры
    sb=st.sidebar
    sb.header("Фильтрация")
    cats=sorted(df['Категория'].unique()); sel_cats=sb.multiselect("Категории",cats,default=cats)
    grps=sorted(df.loc[df['Категория'].isin(sel_cats),'Группа'].unique()); sel_grps=sb.multiselect("Группы",grps,default=grps)
    fmts=sorted(df['Формат'].unique()); sel_fmts=sb.multiselect("Форматы ТТ",fmts,default=fmts)
    whs=sorted(df['Склад'].unique()); sel_whs=sb.multiselect("Склады",whs,default=whs)
    df=df[df['Категория'].isin(sel_cats)&df['Группа'].isin(sel_grps)&df['Формат'].isin(sel_fmts)&df['Склад'].isin(sel_whs)]
    if df.empty: st.warning("Нет данных после фильтров"); return

    # Выручка
    smin,smax=int(df['Продажа с ЗЦ сумма'].min()),int(df['Продажа с ЗЦ сумма'].max())
    srange=sb.slider("Выручка (₽)",smin,smax,(smin,smax),step=1)
    df=df[df['Продажа с ЗЦ сумма'].between(*srange)]

    # Чувствительность и пороги
    sb.header("Чувствительность")
    preset=sb.radio("Пресет",["Нет","Слабая","Средняя","Высокая"])
        defs = {
        "Слабая": ((0.5, 15.0), (5.0, 85.0), 15.0, 60.0),
        "Средняя": ((0.5, 8.0), (10.0, 75.0), 20.0, 80.0),
        "Высокая": ((0.5, 5.0), (20.0, 60.0), 25.0, 90.0),
        "Нет": ((0.0, 100.0), (0.0, 100.0), 0.0, 0.0)
    }
    lw_def,lf_def,hw_def,hf_def=defs[preset]
    # Низкие
    sb.subheader("Низкие списания +\nнизкое закрытие")
    lw=sb.slider("Списания % (диапазон)",0.,100.,lw_def,step=0.1)
    lw_min,lw_max=sb.number_input("Мин. списания",0.,100.,value=lw[0],step=0.1),sb.number_input("Макс. списания",0.,100.,value=lw[1],step=0.1)
    lf=sb.slider("Закрытие % (диапазон)",0.,100.,lf_def,step=0.1)
    lf_min,lf_max=sb.number_input("Мин. закрытие",0.,100.,value=lf[0],step=0.1),sb.number_input("Макс. закрытие",0.,100.,value=lf[1],step=0.1)
    # Высокие
    sb.subheader("Высокие списания +\nвысокое закрытие")
    hw_thr=sb.slider("Порог списания %",0.,200.,hw_def,step=0.1)
    hf_thr=sb.slider("Порог закрытия %",0.,200.,hf_def,step=0.1)

    # Отбор аномалий
    low_df=df[df['Списания %'].between(lw_min,lw_max)&df['Закрытие потребности %'].between(lf_min,lf_max)]
    high_df=df[(df['Списания %']>=hw_thr)&(df['Закрытие потребности %']>=hf_thr)]

    display_anomaly_table(low_df.sort_values('combined_score',ascending=False),"Низкие списания + низкое закрытие")
    display_anomaly_table(high_df.sort_values('combined_score',ascending=False),"Высокие списания + высокое закрытие")

    # График
    mask=df.index.isin(pd.concat([low_df,high_df]).index)
    df_plot=df.copy();df_plot['Статус']=np.where(mask,'Аномалия','Норма')
    fig=px.scatter(df_plot,x='Списания %',y='Закрытие потребности %',color='Статус',size='Продажа с ЗЦ сумма',opacity=0.6,hover_data=['Name_tov','Группа','Формат','Склад'],color_discrete_map={'Норма':'lightgrey','Аномалия':'crimson'})
    st.plotly_chart(fig,use_container_width=True)

    # Экспорт
    buf=BytesIO()
    with pd.ExcelWriter(buf,engine='openpyxl') as w:
        low_df.to_excel(w,sheet_name='Низкие',index=False)
        high_df.to_excel(w,sheet_name='Высокие',index=False)
    buf.seek(0)
    st.download_button("Скачать Excel",buf,"anomalies.xlsx","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

if __name__=="__main__": main()
