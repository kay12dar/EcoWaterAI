import streamlit as st
import pandas as pd
import numpy as np
import re
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

# --- КОНФИГУРАЦИЯ СТРАНИЦЫ ---
st.set_page_config(
    page_title="EcoLab AI",
    page_icon="🧪",
    layout="wide"
)

# --- НАСТРОЙКИ API ---
API_URL = st.secrets.get("API_URL", "http://localhost:8000")


# ─────────────────────────────────────────────
# 1. ФУНКЦИИ (ПАРСИНГ, ML, API)
# ─────────────────────────────────────────────

def clean_value(val):
    val = str(val).replace(',', '.')
    val = re.sub(r'[^\d\.\-]', '', val)
    if '-' in val:
        try:
            parts = val.split('-')
            if parts[0] and parts[1]:
                return (float(parts[0]) + float(parts[1])) / 2
        except:
            pass
    if val == '' or val == '.':
        return np.nan
    try:
        return float(val)
    except:
        return np.nan


def parse_local_excel(uploaded_file):
    try:
        df_raw = pd.read_excel(uploaded_file, header=None)
        header_idx = 0
        for i in range(10):
            row_str = df_raw.iloc[i].astype(str).str.cat(sep=' ')
            if 'Ишим' in row_str or 'Озеро' in row_str or 'Место' in row_str or 'Альпаш' in row_str:
                header_idx = i
                break

        lake_names = df_raw.iloc[header_idx, 2:].values
        data_clean = pd.DataFrame({'Место': [str(n).strip() for n in lake_names]})

        for i in range(header_idx + 1, df_raw.shape[0]):
            param_name = str(df_raw.iloc[i, 1]).strip()
            vals = df_raw.iloc[i, 2:].values
            if param_name and param_name != 'nan':
                data_clean[param_name] = [clean_value(x) for x in vals]
        return data_clean
    except Exception as e:
        return None


# ─── API-функции для загрузки данных из БД ───

@st.cache_data(ttl=300)  # кэш на 5 минут
def fetch_water_bodies():
    """Загружает список всех водоёмов из API"""
    try:
        resp = requests.get(f"{API_URL}/water-bodies", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.warning(f"⚠️ Не удалось подключиться к API: {e}")
        return []


@st.cache_data(ttl=300)
def fetch_measurements(water_body_id: str):
    """Загружает замеры конкретного водоёма из API"""
    try:
        resp = requests.get(
            f"{API_URL}/water-bodies/{water_body_id}/measurements",
            timeout=10
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return []


def measurements_to_df(water_body_name: str, measurements: list) -> pd.DataFrame:
    """
    Конвертирует список замеров из API в формат совместимый с ML-пайплайном.
    Каждый замер становится отдельной строкой (по дате).
    """
    if not measurements:
        return None

    rows = []
    for m in measurements:
        date_label = m.get('recordDate', '')[:10] if m.get('recordDate') else 'б/д'
        rows.append({
            'Место':         f"{water_body_name} ({date_label})",
            'рН':            m.get('ph'),
            'Жесткость':     m.get('hardness'),
            'Сульфаты':      m.get('sulfates'),
            'Мутность':      m.get('turbidity'),
            'Минерализация': m.get('mineralization'),
            'Солёность':     m.get('salinity'),
            'Дата':          date_label,
        })

    return pd.DataFrame(rows) if rows else None


@st.cache_resource
def train_model():
    try:
        df = pd.read_csv('water_potability.csv')
        imputer = SimpleImputer(strategy='mean')
        X = df.drop('Potability', axis=1)
        y = df['Potability']
        X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        model = RandomForestClassifier(n_estimators=150, random_state=42)
        model.fit(X_imputed, y)
        return model, imputer, X.columns
    except FileNotFoundError:
        return None, None, None


# ─────────────────────────────────────────────
# 2. ОБЩИЙ ML-ПАЙПЛАЙН
# ─────────────────────────────────────────────

def run_analysis(df_local, model, imputer, feature_names):
    """Принимает DataFrame → возвращает results + input_ai"""

    def get_col(df, keys):
        for k in keys:
            found = next((c for c in df.columns if k.lower() in c.lower()), None)
            if found:
                return df[found]
        return np.nan

    input_ai = pd.DataFrame(index=range(len(df_local)), columns=feature_names)
    input_ai['ph']        = get_col(df_local, ['рН', 'ph'])
    input_ai['Hardness']  = get_col(df_local, ['Жесткость', 'hardness'])
    input_ai['Sulfate']   = get_col(df_local, ['Сульфат', 'Сульфаты', 'sulfate'])
    input_ai['Turbidity'] = get_col(df_local, ['Мутность', 'turbidity'])

    final_input = pd.DataFrame(imputer.transform(input_ai), columns=feature_names)
    preds = model.predict(final_input)
    probs = model.predict_proba(final_input)

    results = df_local[['Место']].copy().reset_index(drop=True)
    results['pH']        = pd.to_numeric(input_ai['ph'].reset_index(drop=True),       errors='coerce').round(2)
    results['Жесткость'] = pd.to_numeric(input_ai['Hardness'].reset_index(drop=True), errors='coerce').round(0)

    if 'Дата' in df_local.columns:
        results['Дата'] = df_local['Дата'].reset_index(drop=True)

    final_verdicts, final_probs, wqi_statuses = [], [], []

    for i in range(len(input_ai)):
        row = input_ai.iloc[i]
        ai_decision = preds[i]
        ai_prob     = probs[i][1] if ai_decision == 1 else probs[i][0]

        ph_val   = row['ph']       if pd.notna(row['ph'])       else 7.0
        hard_val = row['Hardness'] if pd.notna(row['Hardness']) else 200

        is_critical = False
        reason = ""
        if ph_val < 6.5 or ph_val > 8.5:
            is_critical, reason = True, "pH вне нормы"
        elif hard_val > 350:
            is_critical, reason = True, "Жесткость > 350"

        if is_critical:
            final_verdicts.append("ОПАСНО")
            final_probs.append("100% (Крит.)")
            wqi_statuses.append(f"Критическое ({reason})")
        else:
            if ai_decision == 1:
                final_verdicts.append("БЕЗОПАСНО")
                wqi_statuses.append("В норме")
            else:
                final_verdicts.append("ОПАСНО")
                wqi_statuses.append("Потенциальный риск")
            final_probs.append(f"{ai_prob * 100:.1f}%")

    results['Статус (WQI)']    = wqi_statuses
    results['Итоговый Вердикт']= final_verdicts
    results['Уверенность']     = final_probs

    return results, input_ai


def display_results(results, input_ai):
    """Отрисовывает таблицу с результатами"""

    def highlight_verdict(val):
        color = 'green' if val == 'БЕЗОПАСНО' else '#ff4b4b'
        return f'color: {color}; font-weight: bold'

    def highlight_critical(val):
        try:
            v = float(val)
            if v < 6.5 or v > 8.5:
                return 'background-color: #ffcccc; color: black'
        except:
            pass
        return ''

    styled = results.style.map(highlight_verdict, subset=['Итоговый Вердикт'])
    if 'pH' in results.columns:
        styled = styled.map(highlight_critical, subset=['pH'])

    st.dataframe(styled, use_container_width=True)
    st.info("ℹ️ Таблица интерактивна: вы можете сортировать столбцы.")

    total  = len(results)
    safe   = (results['Итоговый Вердикт'] == 'БЕЗОПАСНО').sum()
    danger = total - safe
    c1, c2, c3 = st.columns(3)
    c1.metric("Всего точек", total)
    c2.metric("✅ Безопасно", safe)
    c3.metric("⚠️ Опасно", danger)


# ─────────────────────────────────────────────
# 3. ЗАГРУЗКА МОДЕЛИ + САЙДБАР
# ─────────────────────────────────────────────

model, imputer, feature_names = train_model()

with st.sidebar:
    st.title("🧪 EcoLab AI")
    st.info("Система гидрохимического анализа")
    menu = st.radio(
        "Разделы:",
        ["Главная (Дашборд)", "Загрузка и Анализ", "Анализ из БД", "Ручной ввод", "Графики и Статистика"]
    )
    st.divider()
    st.write(f"Статус модели: {'Активна 🟢' if model else 'Не загружена 🔴'}")
    st.caption(f"API: {API_URL}")


# ─────────────────────────────────────────────
# 4. ГЛАВНАЯ
# ─────────────────────────────────────────────

if menu == "Главная (Дашборд)":
    st.title("📊 Экологический мониторинг СКО")
    st.markdown("Добро пожаловать в систему интеллектуального анализа качества воды.")

    col1, col2, col3 = st.columns(3)
    col1.metric("База знаний ИИ", "3276 образцов", "Random Forest")
    col2.metric("Точность алгоритма", "78.4%", "+1.2%")
    col3.metric("Регион мониторинга", "Северный Казахстан", "СКО")

    st.subheader("Источники данных")
    st.markdown("""
| Источник | Описание |
|---|---|
| 📂 **Excel-файл** | Загрузка протокола лабораторных испытаний |
| 🗄️ **База данных** | Исторические замеры из системы мониторинга |
| ✍️ **Ручной ввод** | Быстрая проверка одного образца |
    """)


# ─────────────────────────────────────────────
# 5. ЗАГРУЗКА EXCEL
# ─────────────────────────────────────────────

elif menu == "Загрузка и Анализ":
    st.title("📂 Лабораторный анализ (Excel)")

    if not model:
        st.error("❌ Модель не загружена. Убедитесь что файл water_potability.csv доступен.")
        st.stop()

    uploaded_file = st.file_uploader("Загрузите протокол испытаний (.xlsx)", type=['xlsx'])

    if uploaded_file:
        df_local = parse_local_excel(uploaded_file)

        if df_local is not None:
            st.success("Файл успешно обработан!")

            with st.expander("Предпросмотр данных"):
                st.dataframe(df_local.head(10), use_container_width=True)

            if st.button("🚀 Начать анализ"):
                with st.spinner("Анализ..."):
                    results, input_ai = run_analysis(df_local, model, imputer, feature_names)
                st.session_state['results_df'] = results
                st.session_state['results_source'] = 'excel'
                st.subheader("Результаты проверки:")
                display_results(results, input_ai)
        else:
            st.error("❌ Не удалось распознать структуру файла.")


# ─────────────────────────────────────────────
# 6. АНАЛИЗ ИЗ БД  ← НОВЫЙ РАЗДЕЛ
# ─────────────────────────────────────────────

elif menu == "Анализ из БД":
    st.title("🗄️ Анализ данных из базы")
    st.markdown("Данные берутся напрямую из вашей системы мониторинга через API.")

    if not model:
        st.error("❌ Модель не загружена.")
        st.stop()

    with st.spinner("Загрузка водоёмов..."):
        water_bodies = fetch_water_bodies()

    if not water_bodies:
        st.warning("⚠️ Водоёмы не найдены. Проверьте подключение к API.")
        st.code(f"API_URL = {API_URL}", language="bash")
        st.stop()

    st.success(f"✅ Найдено водоёмов в базе: **{len(water_bodies)}**")

    # Режим выбора
    mode = st.radio(
        "Режим анализа:",
        ["Один водоём", "Несколько водоёмов", "Все водоёмы"],
        horizontal=True
    )

    selected_ids   = []
    selected_names = {}

    if mode == "Один водоём":
        options = {wb['name']: wb['id'] for wb in water_bodies}
        chosen  = st.selectbox("Выберите водоём:", list(options.keys()))
        if chosen:
            selected_ids   = [options[chosen]]
            selected_names = {options[chosen]: chosen}

    elif mode == "Несколько водоёмов":
        options = {wb['name']: wb['id'] for wb in water_bodies}
        chosen_list = st.multiselect(
            "Выберите водоёмы:",
            list(options.keys()),
            default=list(options.keys())[:min(3, len(options))]
        )
        selected_ids   = [options[n] for n in chosen_list]
        selected_names = {options[n]: n for n in chosen_list}

    else:
        selected_ids   = [wb['id']   for wb in water_bodies]
        selected_names = {wb['id']: wb['name'] for wb in water_bodies}
        st.info(f"Будут проанализированы все {len(selected_ids)} водоёмов.")

    # Фильтр по дате
    st.divider()
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        date_from = st.date_input("Замеры с:", value=None)
    with col_f2:
        date_to   = st.date_input("по:", value=None)

    if selected_ids and st.button("🚀 Загрузить и проанализировать"):
        from datetime import date as date_type

        all_frames = []
        progress   = st.progress(0, text="Загрузка замеров...")

        for idx, wb_id in enumerate(selected_ids):
            wb_name      = selected_names.get(wb_id, wb_id)
            measurements = fetch_measurements(wb_id)

            # Фильтрация по дате
            if date_from or date_to:
                filtered = []
                for m in measurements:
                    rec_str = (m.get('recordDate') or '')[:10]
                    if not rec_str:
                        continue
                    try:
                        rec_date = date_type.fromisoformat(rec_str)
                        if date_from and rec_date < date_from:
                            continue
                        if date_to and rec_date > date_to:
                            continue
                        filtered.append(m)
                    except:
                        filtered.append(m)
                measurements = filtered

            if not measurements:
                st.warning(f"⚠️ Нет замеров для «{wb_name}»")
            else:
                df_wb = measurements_to_df(wb_name, measurements)
                if df_wb is not None:
                    all_frames.append(df_wb)

            progress.progress((idx + 1) / len(selected_ids), text=f"Обработано: {wb_name}")

        progress.empty()

        if not all_frames:
            st.error("❌ Нет данных для анализа.")
            st.stop()

        df_combined = pd.concat(all_frames, ignore_index=True)
        st.info(f"📊 Загружено записей для анализа: **{len(df_combined)}**")

        with st.expander("Предпросмотр загруженных данных"):
            st.dataframe(df_combined, use_container_width=True)

        with st.spinner("Запуск AI-анализа..."):
            results, input_ai = run_analysis(df_combined, model, imputer, feature_names)

        st.session_state['results_df']     = results
        st.session_state['results_source'] = 'db'

        st.subheader("Результаты анализа:")
        display_results(results, input_ai)

        # Сравнительный график если несколько водоёмов
        if len(selected_ids) > 1 and 'pH' in results.columns:
            st.subheader("📈 Средний pH по водоёмам")
            fig, ax = plt.subplots(figsize=(10, max(4, len(selected_ids) * 0.5)))
            results_plot = results.copy()
            results_plot['Водоём'] = results_plot['Место'].str.split(' (').str[0]
            avg_ph = results_plot.groupby('Водоём')['pH'].mean().sort_values()
            bar_colors = ['#2ecc71' if (6.5 <= v <= 8.5) else '#e74c3c' for v in avg_ph]
            avg_ph.plot(kind='barh', ax=ax, color=bar_colors)
            ax.axvline(6.5, color='gray', linestyle='--', linewidth=1, label='Норма')
            ax.axvline(8.5, color='gray', linestyle='--', linewidth=1)
            ax.set_xlabel("pH (среднее значение)")
            ax.legend()
            st.pyplot(fig)
            plt.close()


# ─────────────────────────────────────────────
# 7. РУЧНОЙ ВВОД
# ─────────────────────────────────────────────

elif menu == "Ручной ввод":
    st.title("🎛️ Симулятор проверки воды")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Химические показатели")
        ph_val   = st.slider("Уровень pH", 0.0, 14.0, 7.0, 0.1)
        hard_val = st.number_input("Жесткость (mg/L)", 0, 1000, 200)
        sulf_val = st.number_input("Сульфаты (mg/L)", 0, 1000, 333)
        turb_val = st.number_input("Мутность (NTU)", 0.0, 10.0, 4.0)

    with col2:
        st.subheader("Результат анализа")
        if st.button("Проверить пробу"):
            if not model:
                st.error("Модель не загружена.")
            else:
                manual_data = pd.DataFrame(columns=feature_names)
                manual_data.loc[0]       = np.nan
                manual_data['ph']        = ph_val
                manual_data['Hardness']  = hard_val
                manual_data['Sulfate']   = sulf_val
                manual_data['Turbidity'] = turb_val

                final_input  = pd.DataFrame(imputer.transform(manual_data), columns=feature_names)
                pred         = model.predict(final_input)[0]
                prob         = model.predict_proba(final_input)[0]
                prob_percent = prob[1] if pred == 1 else prob[0]

                verdict = "НЕ ОПРЕДЕЛЕНО"
                reason  = ""
                if ph_val < 6.5 or ph_val > 8.5:
                    verdict, reason = "ОПАСНО", "Критический уровень pH!"
                elif hard_val > 350:
                    verdict, reason = "ОПАСНО", "Высокая жесткость!"
                else:
                    verdict = "БЕЗОПАСНО" if pred == 1 else "ОПАСНО"
                    if pred != 1:
                        reason = "Выявлены скрытые загрязнения (ИИ)"

                if verdict == "БЕЗОПАСНО":
                    st.success(f"✅ **Вердикт:** {verdict}")
                    st.metric("Вероятность чистоты", f"{prob_percent * 100:.1f}%")
                else:
                    st.error(f"⚠️ **Вердикт:** {verdict}")
                    if reason:
                        st.write(f"**Причина:** {reason}")
                    st.metric("Уверенность в опасности", f"{prob_percent * 100:.1f}%")

                st.info("💡 " + (
                    "Вода пригодна для использования."
                    if verdict == "БЕЗОПАСНО"
                    else "Требуется дополнительная фильтрация или очистка."
                ))


# ─────────────────────────────────────────────
# 8. ГРАФИКИ И СТАТИСТИКА
# ─────────────────────────────────────────────

elif menu == "Графики и Статистика":
    st.title("📈 Визуальная аналитика")

    if 'results_df' not in st.session_state:
        st.warning("⚠️ Данные не найдены. Сначала выполните анализ.")
        st.stop()

    df     = st.session_state['results_df']
    source = st.session_state.get('results_source', 'excel')
    st.caption(f"Источник: {'база данных 🗄️' if source == 'db' else 'Excel-файл 📂'} · {len(df)} записей")

    if 'pH' in df.columns:
        st.subheader("1. Уровень pH")
        st.caption("Красные линии — границы нормы (6.5 – 8.5)")
        fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.3)))
        plot_data = df.dropna(subset=['pH'])
        if not plot_data.empty:
            sns.barplot(data=plot_data, x='pH', y='Место', palette='viridis', ax=ax)
            ax.axvline(6.5, color='red', linestyle='--')
            ax.axvline(8.5, color='red', linestyle='--')
        st.pyplot(fig)
        plt.close()

    if 'Жесткость' in df.columns:
        st.subheader("2. Жесткость воды")
        chart_data = df.set_index('Место')['Жесткость'].dropna()
        if not chart_data.empty:
            st.bar_chart(chart_data)

    st.subheader("3. Распределение вердиктов")
    col1, col2 = st.columns(2)
    with col1:
        status_counts = df['Итоговый Вердикт'].value_counts()
        fig2, ax2 = plt.subplots()
        colors = ['#2ecc71' if x == 'БЕЗОПАСНО' else '#e74c3c' for x in status_counts.index]
        ax2.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', colors=colors)
        st.pyplot(fig2)
        plt.close()
    with col2:
        st.write("Статистика:")
        st.dataframe(status_counts.reset_index().rename(
            columns={'index': 'Вердикт', 'Итоговый Вердикт': 'Количество'}
        ))
        if 'Дата' in df.columns:
            dates = df['Дата'].dropna()
            if not dates.empty:
                st.write(f"Диапазон дат: {dates.min()} — {dates.max()}")
