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
    df['Оценка аномалии'] = -model.decision_function(X)  # higher = more anomalous
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

def display_subcategory_impact(df, title):
    st.subheader(f"Влияние на подкатегории: {title}")
    total_sales_by_group = df.groupby('Группа')['Продажа с ЗЦ сумма'].sum()
    impact = (
        df.groupby('Группа').apply(
            lambda sub: pd.Series({
                'Количество позиций': len(sub),
                'Сумма продаж аномалии': sub['Продажа с ЗЦ сумма'].sum(),
                'Доля продаж (%)': sub['Продажа с ЗЦ сумма'].sum() / total_sales_by_group[sub.name] * 100,
                'Средняя оценка аномалии': sub['Оценка аномалии'].mean(),
                'Взв. средняя оценка': np.average(sub['Оценка аномалии'], weights=sub['Продажа с ЗЦ сумма']),
                'Суммарный комб. скор': sub['Комбинированный скор'].sum()
            })
        )
        .reset_index()
    )
    st.dataframe(impact.style.format({
        'Сумма продаж аномалии': '{:.0f}',
        'Доля продаж (%)': '{:.1f}',
        'Средняя оценка аномалии': '{:.3f}',
        'Взв. средняя оценка': '{:.3f}',
        'Суммарный комб. скор': '{:.2f}'
    }), use_container_width=True)

def main():
    st.set_page_config(page_title="Аномалии", layout="wide")
    st.title("Анализ аномалий: списания и закрытие потребности")

    uploaded = st.file_uploader("Загрузите CSV-файл", type="csv")
    if not uploaded:
        st.info("Пожалуйста, загрузите файл для анализа")
        return

    df = load_data(uploaded)
    df = compute_scores(df)

    low_cond = df['Списания %'].between(0.5, 8) & df['Закрытие потребности %'].between(10, 75)
    high_cond = (df['Списания %'] >= 20) & (df['Закрытие потребности %'] >= 80)

    low_df = filter_sort(df, low_cond)
    high_df = filter_sort(df, high_cond)

    display_section(low_df, "Низкие списания + Низкое закрытие потребности", 'Greens', (0.5, 8), (10, 75))
    display_section(high_df, "Высокие списания + Высокое закрытие потребности", 'Reds', (20, df['Списания %'].max()), (80, df['Закрытие потребности %'].max()))

    # вкладка влияния на подкатегории для обеих групп
    display_subcategory_impact(low_df, "Низкие списания + Низкое закрытие")
    display_subcategory_impact(high_df, "Высокие списания + Высокое закрытие")

    # Донат-диаграмма
    total = len(df)
    counts = {"Низкие": len(low_df), "Высокие": len(high_df), "Остальные": total - len(low_df) - len(high_df)}
    fig = px.pie(names=list(counts.keys()), values=list(counts.values()), hole=0.4)
    st.plotly_chart(fig, use_container_width=True)

    # Скачивание
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        low_df.to_excel(writer, sheet_name='Низкие аномалии', index=False)
        high_df.to_excel(writer, sheet_name='Высокие аномалии', index=False)
        # сохраняем и влияние на подкатегории
        display_low = low_df[['Группа','Комбинированный скор','Продажа с ЗЦ сумма']]
        display_high = high_df[['Группа','Комбинированный скор','Продажа с ЗЦ сумма']]
        impact_low = low_df.groupby('Группа').apply(lambda sub: sub['Комбинированный скор'].sum()).reset_index(name='Суммарный комб. скор')
        impact_high = high_df.groupby('Группа').apply(lambda sub: sub['Комбинированный скор'].sum()).reset_index(name='Суммарный комб. скор')
        impact_low.to_excel(writer, sheet_name='Влияние низких', index=False)
        impact_high.to_excel(writer, sheet_name='Влияние высоких', index=False)
    buf.seek(0)
    st.download_button(
        "Скачать отчет XLSX",
        data=buf,
        file_name="anomalies_full_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

if __name__ == "__main__":
    main()
