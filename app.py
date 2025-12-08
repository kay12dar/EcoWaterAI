import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix

# --- НАСТРОЙКА СТРАНИЦЫ ---
st.set_page_config(
    page_title="EcoWater AI",
    page_icon="💧",
    layout="wide"
)


# --- 1. ФУНКЦИИ ОЧИСТКИ ДАННЫХ ---
def clean_value(val):
    """Превращает грязные строки Excel в числа"""
    val = str(val).replace(',', '.')
    val = re.sub(r'[^\d\.\-]', '', val)
    if '-' in val:
        try:
            parts = val.split('-')
            if parts[0] and parts[1]:
                return (float(parts[0]) + float(parts[1])) / 2
        except:
            pass
    if val == '' or val == '.': return np.nan
    try:
        return float(val)
    except:
        return np.nan


def parse_local_excel(uploaded_file):
    """Читает и транспонирует специфичный Excel файл пользователя"""
    try:
        df_raw = pd.read_excel(uploaded_file, header=None)

        # Ищем строку заголовка (где названия озер)
        header_idx = 0
        for i in range(10):
            row_str = df_raw.iloc[i].astype(str).str.cat(sep=' ')
            if 'Ишим' in row_str or 'Озеро' in row_str or 'Место' in row_str or 'Альпаш' in row_str:
                header_idx = i
                break

        # Формируем таблицу
        lake_names = df_raw.iloc[header_idx, 2:].values
        data_clean = pd.DataFrame({'Место': [str(n).strip() for n in lake_names]})

        # Парсим параметры
        for i in range(header_idx + 1, df_raw.shape[0]):
            param_name = str(df_raw.iloc[i, 1]).strip()
            vals = df_raw.iloc[i, 2:].values
            if param_name and param_name != 'nan':
                data_clean[param_name] = [clean_value(x) for x in vals]

        return data_clean
    except Exception as e:
        st.error(f"Ошибка при разборе файла: {e}")
        return None


# --- 2. ОБУЧЕНИЕ МОДЕЛИ (Кэшируем, чтобы не учить каждый раз заново) ---
@st.cache_resource
def train_model():
    try:
        df = pd.read_csv('water_potability.csv')

        # Заполнение пропусков
        imputer = SimpleImputer(strategy='mean')
        X = df.drop('Potability', axis=1)
        y = df['Potability']

        X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

        # Разделение и обучение
        X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        acc = accuracy_score(y_test, model.predict(X_test))

        return model, imputer, acc, X.columns
    except FileNotFoundError:
        return None, None, None, None


# --- ИНТЕРФЕЙС ПРИЛОЖЕНИЯ ---

# Сайдбар (Информация)
with st.sidebar:
    st.title("⚙️ Панель управления")
    st.info(
        "Экзаменационный проект\n\n**Тема:** Интеллектуальный модуль оценки загрязнения озер.")
    st.divider()
    st.write("Используемый алгоритм:")
    st.code("RandomForestClassifier")

# Заголовок
st.title("💧 Система ИИ: Оценка качества воды")
st.markdown(
    "Модуль анализирует химический состав воды и определяет степень её загрязнения, используя модель машинного обучения, обученную на международных данных.")

# --- БЛОК 1: СТАТУС МОДЕЛИ ---
st.header("1. Состояние Модели")
model, imputer, accuracy, feature_names = train_model()

if model is None:
    st.error("❌ Ошибка: Файл 'water_potability.csv' не найден! Загрузите датасет в папку проекта.")
else:
    col1, col2, col3 = st.columns(3)
    col1.metric("Статус обучения", "Обучена ✅")
    col2.metric("Точность (Accuracy)", f"{accuracy * 100:.1f}%")
    col3.metric("Размер базы знаний", "3276 примеров")

    # График важности признаков (Критерий "Анализ")
    with st.expander("📊 Посмотреть, что важно для ИИ (Важность признаков)"):
        importances = pd.DataFrame({
            'Признак': feature_names,
            'Важность': model.feature_importances_
        }).sort_values(by='Важность', ascending=False)

        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(data=importances, x='Важность', y='Признак', ax=ax, palette='viridis')
        st.pyplot(fig)

# --- БЛОК 2: ЗАГРУЗКА ДАННЫХ ---
st.header("2. Загрузка локальных данных")
st.markdown("Загрузите Excel-файл с результатами экспресс-теста (например, пробы из реки Ишим).")

uploaded_file = st.file_uploader("Выберите файл .xlsx", type="xlsx")

if uploaded_file is not None:
    # Парсинг
    df_local = parse_local_excel(uploaded_file)

    if df_local is not None:
        st.write("✅ **Распознанные данные:**")
        st.dataframe(df_local)

        # --- БЛОК 3: АНАЛИЗ ---
        st.header("3. Результаты анализа")

        if st.button("🚀 Запустить анализ загрязнения"):
            with st.spinner('ИИ анализирует пробы...'):
                # Подготовка данных для модели (Mapping)
                input_data = pd.DataFrame(index=range(len(df_local)), columns=feature_names)


                # Сопоставление колонок (Гибкий поиск)
                def get_col(df, key):
                    found = next((c for c in df.columns if key.lower() in c.lower()), None)
                    return df[found] if found else np.nan


                # Заполняем тем, что есть в Excel
                input_data['ph'] = get_col(df_local, 'рН')
                input_data['Hardness'] = get_col(df_local, 'Жесткость')
                input_data['Sulfate'] = get_col(df_local, 'Сульфат')
                input_data['Turbidity'] = get_col(df_local, 'Мутность')
                # Если твердых частиц нет, можно пробовать брать Щелочность или Оставлять пустым (заполнит Imputer)

                # Заполняем пропуски средними значениями (из обучения)
                final_input = pd.DataFrame(imputer.transform(input_data), columns=feature_names)

                # Предсказание
                preds = model.predict(final_input)
                probs = model.predict_proba(final_input)

                # Красивый вывод результатов
                results = []
                for i, row in df_local.iterrows():
                    is_safe = preds[i] == 1
                    prob = probs[i][1] if is_safe else probs[i][0]
                    status_text = "ЧИСТО / ПРИГОДНО" if is_safe else "ЗАГРЯЗНЕНО"
                    color = "green" if is_safe else "red"

                    results.append({
                        "Водоем": row['Место'],
                        "Статус": status_text,
                        "Вероятность": f"{prob * 100:.1f}%",
                        "pH": f"{input_data.iloc[i]['ph']:.1f}",
                        "Жесткость": f"{input_data.iloc[i]['Hardness']:.1f}"
                    })

                res_df = pd.DataFrame(results)

                # Отображение карточками или таблицей
                st.subheader("Вердикт системы:")


                # Стилизация таблицы цветом
                def color_status(val):
                    color = '#ffcdd2' if 'ЗАГРЯЗНЕНО' in val else '#c8e6c9'
                    return f'background-color: {color}'


                st.dataframe(res_df.style.applymap(color_status, subset=['Статус']), use_container_width=True)

                # Скачивание отчета
                csv = res_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Скачать отчет (CSV)",
                    data=csv,
                    file_name="report_water_quality.csv",
                    mime="text/csv",
                )

                # Вывод сообщения
                cnt_polluted = len(res_df[res_df['Статус'].str.contains('ЗАГРЯЗНЕНО')])
                if cnt_polluted > 0:
                    st.warning(f"⚠️ Внимание! Обнаружено загрязненных объектов: {cnt_polluted}")
                else:
                    st.success("🎉 Все проверенные водоемы в норме!")