import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from sklearn.ensemble import IsolationForest
import plotly.express as px

def load_data(uploaded):
    df = pd.read_csv(uploaded)
    df.columns = df.columns.str.strip()
    df['Списания %'] = (
        df['ЗЦ2_срок_качество_%']
        .astype(str).str.replace(',', '.').str.rstrip('%').astype(float)
    )
    df['Закрытие потребности %'] = (
        df['Закрытие потребности_%']
        .astype(str).str.replace(',', '.').str.rstrip('%').astype(float)
    )
    df['Продажа с ЗЦ сумма'] = pd.to_numeric(df['Продажа_с_ЗЦ_сумма'], errors='coerce').fillna(0)
    return df.dropna(subset=['Списания %', 'Закрытие потребности %'])

def compute_scores(df):
    X = df[['Списания %', 'Закрытие потребности %']]
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(X)
    scores = model.decision_function(X)
    df['Оценка аномалии'] = -scores  # higher = more anomalous
    df['Комбинированный скор'] = df['Оценка аномалии'] * df['Продажа с ЗЦ сумма']
    return df

def filter_sort(df, condition):
    return df[condition].sort_values('Комбинированный скор', ascending=False)

def display_section(df_sec, title, cmap, waste_range=None, fill_range=None):
    st.subheader(f"{title} (Найдено: {len(df_sec)})")
    # Метрики
    mean_waste = df_sec['Списания %'].mean()
    wavg_waste = np.average(df_sec['Списания %'], weights=df_sec['Продажа с ЗЦ сумма']) if df_sec['Продажа с ЗЦ сумма'].sum() else np.nan
    mean_fill = df_sec['Закрытие потребности %'].mean()
    wavg_fill = np.average(df_sec['Закрытие потребности %'], weights=df_sec['Продажа с ЗЦ сумма']) if df_sec['Продажа с ЗЦ сумма'].sum() else np.nan

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Среднее списания %", f"{mean_waste:.1f}")
    c2.metric("Взв. среднее списания %", f"{wavg_waste:.1f}")
    c3.metric("Среднее закрытие %", f"{mean_fill:.1f}")
    c4.metric("Взв. среднее закрытие %", f"{wavg_fill:.1f}")

    # Стилизация таблицы
    cols = ['Категория', 'Группа', 'Name_tov',
            'Списания %', 'Закрытие потребности %',
            'Продажа с ЗЦ сумма', 'Оценка аномалии', 'Комбинированный скор']
    styler = df_sec[cols].style.format({
        'Списания %': '{:.1f}', 'Закрытие потребности %': '{:.1f}',
        'Продажа с ЗЦ сумма': '{:.0f}',
        'Оценка аномалии': '{:.3f}', 'Комбинированный скор': '{:.2f}'
    })
    if waste_range:
        styler = styler.background_gradient(subset=['Списания %'], cmap=cmap, vmin=waste_range[0], vmax=waste_range[1])
    else:
        styler = styler.background_gradient(subset=['Списания %'], cmap=cmap)
    if fill_range:
        styler = styler.background_gradient(subset=['Закрытие потребности %'], cmap=cmap, vmin=fill_range[0], vmax=fill_range[1])
    else:
        styler = styler.background_gradient(subset=['Закрытие потребности %'], cmap=cmap)
    st.dataframe(styler, use_container_width=True)

def main():
    st.set_page_config(page_title="Аномалии", layout="wide")
    st.title("Анализ аномалий: списания и закрытие потребности")

    uploaded = st.file_uploader("Загрузите CSV-файл", type="csv")
    if not uploaded:
        st.info("Пожалуйста, загрузите файл для анализа")
        return

    df = load_data(uploaded)
    df = compute_scores(df)

    # Условия выбора
    low_cond = df['Списания %'].between(0.5, 8) & df['Закрытие потребности %'].between(10, 75)
    high_cond = (df['Списания %'] >= 20) & (df['Закрытие потребности %'] >= 80)

    low_df = filter_sort(df, low_cond)
    high_df = filter_sort(df, high_cond)

    # Отображение разделов
    display_section(
        low_df,
        "Низкие списания + Низкое закрытие потребности",
        cmap='Greens',
        waste_range=(0.5, 8),
        fill_range=(10, 75)
    )
    display_section(
        high_df,
        "Высокие списания + Высокое закрытие потребности",
        cmap='Reds',
        waste_range=(20, df['Списания %'].max()),
        fill_range=(80, df['Закрытие потребности %'].max())
    )

    # Донат-диаграмма
    total = len(df)
    counts = {
        "Низкие": len(low_df),
        "Высокие": len(high_df),
        "Остальные": total - len(low_df) - len(high_df)
    }
    fig = px.pie(names=list(counts.keys()), values=list(counts.values()), hole=0.4)
    st.plotly_chart(fig, use_container_width=True)

    # Топ-30 по группам
    st.subheader("Топ-30 наиболее критичных позиций по группам")
    df_sorted = df.sort_values('Комбинированный скор', ascending=False)
    top30 = (df_sorted
             .groupby('Группа', as_index=False)
             .head(30)
             [['Группа', 'Name_tov', 'Списания %', 'Закрытие потребности %',
               'Продажа с ЗЦ сумма', 'Комбинированный скор']])
    with st.expander("Развернуть топ-30 по каждой группе"):
        st.dataframe(top30.style.format({
            'Списания %': '{:.1f}',
            'Закрытие потребности %': '{:.1f}',
            'Продажа с ЗЦ сумма': '{:.0f}',
            'Комбинированный скор': '{:.2f}'
        }), use_container_width=True)

    # Скачивание результатов
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        low_df.to_excel(writer, sheet_name='Низкие аномалии', index=False)
        high_df.to_excel(writer, sheet_name='Высокие аномалии', index=False)
        top30.to_excel(writer, sheet_name='Топ-30 позиций', index=False)
    buffer.seek(0)
    st.download_button(
        "Скачать результаты в Excel",
        data=buffer,
        file_name="anomalies_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

if __name__ == "__main__":
    main()

