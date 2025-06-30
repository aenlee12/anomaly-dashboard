# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from sklearn.ensemble import IsolationForest
import plotly.express as px

st.set_page_config(page_title="Аномалии: списания & закрытие", layout="wide")


def load_and_prepare(uploaded):
    name = uploaded.name.lower()
    if name.endswith((".xls", ".xlsx")):
        df = pd.read_excel(uploaded, engine="openpyxl")
    else:
        df = pd.read_csv(uploaded)
    df.columns = df.columns.str.strip()

    # Переименование
    col_map = {}
    if "parent_group_name" in df: col_map["parent_group_name"]="Категория"
    if "group_name" in df:         col_map["group_name"]      ="Группа"
    fmt = next((c for c in df if "формат" in c.lower() or "format" in c.lower()), None)
    if fmt: col_map[fmt]="Формат"
    skl = next((c for c in df if "name_sklad" in c.lower()), None) or \
          next((c for c in df if "sklad" in c.lower()), None)
    if skl: col_map[skl]="Склад"
    df = df.rename(columns=col_map)

    # Проверка
    for req in ["Категория","Группа","Name_tov","Формат","Склад"]:
        if req not in df:
            raise KeyError(f"Missing {req}")

    # Выручка
    sale = next((c for c in df if "продажа" in c.lower() and "сумма" in c.lower()), None)
    df["Продажа с ЗЦ сумма"] = pd.to_numeric(df[sale],errors="coerce").fillna(0)

    # Закрытие %
    fill = next((c for c in df if "закрытие" in c.lower()), None)
    df["Закрытие потребности %"] = (
        pd.to_numeric(df[fill].astype(str)
                      .str.replace(",",".").str.rstrip("%"),
                      errors="coerce")*100
    )

    # Списания %
    zc2  = next((c for c in df if c.lower().startswith("зц2")), None)
    srok = next((c for c in df if c.lower()=="срок"), None)
    kach = next((c for c in df if c.lower()=="качество"), None)
    for c in (zc2,srok,kach):
        df[c]=pd.to_numeric(df[c],errors="coerce").fillna(0)
    df["Списания %"]=np.where(
        df["Продажа с ЗЦ сумма"]>0,
        (df[zc2]+df[srok]+df[kach])/df["Продажа с ЗЦ сумма"]*100,
        0
    )

    df = df.dropna(subset=["Категория","Группа","Name_tov","Формат","Склад",
                           "Списания %","Закрытие потребности %"])
    return df


def score_anomalies(df):
    df = df.copy()
    df["anomaly_score"] = 0.0
    for grp,sub in df.groupby("Группа"):
        if len(sub)<2: continue
        X=sub[["Списания %","Закрытие потребности %"]]
        iso=IsolationForest(contamination=0.05,random_state=42,n_estimators=30,n_jobs=-1)
        iso.fit(X)
        df.loc[sub.index,"anomaly_score"]=-iso.decision_function(X)
    df["severity"]=df["anomaly_score"].abs()
    df["share"]=df["Продажа с ЗЦ сумма"]/df.groupby("Группа")["Продажа с ЗЦ сумма"].transform("sum").fillna(1)
    df["combined"]=df["severity"]*df["share"]
    return df


@st.cache_data
def process_data(uploaded):
    base = load_and_prepare(uploaded)
    return score_anomalies(base)


@st.cache_data
def get_hierarchy_df(full_df):
    return (
        full_df
        .groupby(["Склад","Формат","Категория","Группа"])
        .agg({"Списания %":"mean",
              "Закрытие потребности %":"mean",
              "Продажа с ЗЦ сумма":"sum"})
        .reset_index()
    )


def display_anomaly_table(df,title):
    # Пересчёт group_mean и weighted_mean на текущем df
    df = df.copy()
    df["group_mean_waste"] = df.groupby("Группа")["Списания %"].transform("mean")
    wmap = {
        grp: np.average(
            sub["Списания %"], weights=sub["Продажа с ЗЦ сумма"]
        ) if sub["Продажа с ЗЦ сумма"].sum()>0 else sub["Списания %"].mean()
        for grp,sub in df.groupby("Группа")
    }
    df["group_weighted_mean_waste"] = df["Группа"].map(wmap)

    st.subheader(f"{title} — {len(df)}")
    cols=["Категория","Группа","Формат","Склад","Name_tov",
          "Списания %","group_mean_waste","group_weighted_mean_waste",
          "Закрытие потребности %","Продажа с ЗЦ сумма","severity","combined"]
    rename={
        "group_mean_waste":"Среднее в группе %",
        "group_weighted_mean_waste":"Ср.взв. в группе %",
        "severity":"Степень аномалии",
        "combined":"Скор"
    }
    styled = (
        df[cols]
        .rename(columns=rename)
        .style
        .format({
            "Списания %":"{:.1f}",
            "Среднее в группе %":"{:.1f}",
            "Ср.взв. в группе %":"{:.1f}",
            "Закрытие потребности %":"{:.1f}",
            "Продажа с ЗЦ сумма":"{:.0f}",
            "Степень аномалии":"{:.3f}",
            "Скор":"{:.3f}"
        })
        .background_gradient(subset=["Списания %"], cmap="Reds")
        .background_gradient(subset=["Закрытие потребности %"], cmap="Blues")
        .background_gradient(subset=["Скор"], cmap="Purples")
    )
    st.dataframe(styled,use_container_width=True)


def main():
    st.title("Аномалии: списания и закрытие потребности")
    uploaded = st.file_uploader("Загрузите CSV/Excel", type=["csv","xls","xlsx"])
    if not uploaded:
        return

    full_df = process_data(uploaded)

    # Sidebar: вернуть фильтрацию по складу и формату
    df = full_df.copy()
    sb = st.sidebar
    sb.header("Фильтрация")

    # Склады
    whs = sorted(df["Склад"].unique())
    sel_whs = sb.multiselect("Склады", whs, default=whs)
    df = df[df["Склад"].isin(sel_whs)]

    # Форматы
    fmts = sorted(df["Формат"].unique())
    sel_fmts = sb.multiselect("Форматы", fmts, default=fmts)
    df = df[df["Формат"].isin(sel_fmts)]

    # Категории
    cats = sorted(df["Категория"].unique())
    sel_cats = sb.multiselect("Категории", cats, default=cats)
    df = df[df["Категория"].isin(sel_cats)]

    # Группы
    grps = sorted(df["Группа"].unique())
    sel_grps = sb.multiselect("Группы", grps, default=grps)
    df = df[df["Группа"].isin(sel_grps)]

    # Выручка
    smin,smax = int(df["Продажа с ЗЦ сумма"].min()), int(df["Продажа с ЗЦ сумма"].max())
    rng = sb.slider("Выручка (₽)", smin, smax, (smin,smax))
    df = df[df["Продажа с ЗЦ сумма"].between(*rng)]

    # Чувствительность
    sb.markdown("---")
    sb.header("Чувствительность")
    presets = {
        "Нет":     ((0,100),(0,100),0,0),
        "Слабая":  ((0.5,15),(5,85),15,60),
        "Средняя": ((0.5,8),(10,75),20,80),
        "Высокая": ((0.5,5),(20,60),25,90),
    }
    p = sb.radio("Пресет", list(presets), index=2)
    lw,lf,hw,hf = presets[p]

    sb.subheader("Низкие списания + низкое закрытие")
    low_min,low_max = sb.slider("Списания %",0,100,lw,step=0.1)
    close_min,close_max = sb.slider("Закрытие %",0,100,lf,step=0.1)

    sb.markdown("---")
    sb.subheader("Высокие списания + высокое закрытие")
    hw_thr = sb.number_input("Порог списания %",0,200,hw,step=0.1)
    hf_thr = sb.number_input("Порог закрытия %",0,200,hf,step=0.1)

    low_df  = df[df["Списания %"].between(low_min,low_max) &
                 df["Закрытие потребности %"].between(close_min,close_max)]
    high_df = df[(df["Списания %"]>=hw_thr)&(df["Закрытие потребности %"]>=hf_thr)]

    # Аномалии
    display_anomaly_table(low_df,"Низкие списания + низкое закрытие (топ-100)")
    display_anomaly_table(high_df,"Высокие списания + высокое закрытие (топ-100)")

    # Пузырьковая
    mask_anom   = df.index.isin(pd.concat([low_df,high_df]).index)
    mask_report = df.index.isin(
        pd.concat([low_df.head(100),high_df.head(100)]).index
    )
    df_plot = df.copy()
    df_plot["Статус"] = np.where(mask_report,"В отчете",
                           np.where(mask_anom,"Аномалия","Норма"))
    fig = px.scatter(
        df_plot, x="Закрытие потребности %", y="Списания %",
        color="Статус", size="Продажа с ЗЦ сумма", opacity=0.6,
        hover_data=["Name_tov","Группа","Формат","Склад"],
        color_discrete_map={"Норма":"lightgray","Аномалия":"crimson","В отчете":"purple"}
    )
    fig.update_xaxes(range=[0,150])
    fig.update_yaxes(range=[0,150])
    st.subheader("Диаграмма всех SKU")
    st.plotly_chart(fig, use_container_width=True)

    # Иерархическая фильтрация: каскад multiselect по full_df
    comp = get_hierarchy_df(full_df)
    base = full_df.copy()
    st.subheader("Иерархическая фильтрация")
    # Склады → Форматы → Категории → Группы
    wh_sel = st.multiselect("Склады", sorted(base["Склад"].unique()), default=sorted(base["Склад"].unique()))
    base = base[base["Склад"].isin(wh_sel)]
    fmt_sel= st.multiselect("Форматы", sorted(base["Формат"].unique()), default=sorted(base["Формат"].unique()))
    base = base[base["Формат"].isin(fmt_sel)]
    cat_sel= st.multiselect("Категории", sorted(base["Категория"].unique()), default=sorted(base["Категория"].unique()))
    base = base[base["Категория"].isin(cat_sel)]
    grp_sel= st.multiselect("Группы", sorted(base["Группа"].unique()), default=sorted(base["Группа"].unique()))
    base = base[base["Группа"].isin(grp_sel)]

    # Отображаем агрегат comp отфильтрованный по этим выборам
    hdf = comp[
        comp["Склад"].isin(wh_sel)&
        comp["Формат"].isin(fmt_sel)&
        comp["Категория"].isin(cat_sel)&
        comp["Группа"].isin(grp_sel)
    ]
    st.dataframe(
        hdf[["Склад","Формат","Категория","Группа","Списания %",
             "Закрытие потребности %","Продажа с ЗЦ сумма"]],
        use_container_width=True
    )

    # Экспорт
    buf=BytesIO()
    with pd.ExcelWriter(buf,engine="openpyxl") as w:
        low_df.to_excel(w,sheet_name="Низкие",index=False)
        high_df.to_excel(w,sheet_name="Высокие",index=False)
    buf.seek(0)
    st.download_button("Скачать Excel",buf,
                       "anomalies.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


if __name__=="__main__":
    main()
