import os
import re

# Укажи путь к папке с файлами
positive_path = r"C:\Users\Admin\Desktop\Intelligent Systems\Assessment 3\NLP-Project\positive.review"
negative_path = r"C:\Users\Admin\Desktop\Intelligent Systems\Assessment 3\NLP-Project\negative.review"

def extract_reviews(file_path):
    """Извлекает текст отзывов из файла .review"""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
        
    # Найдём блоки <review>...</review>
    review_blocks = re.findall(r"<review>(.*?)</review>", content, re.DOTALL)
    
    reviews = []
    for block in review_blocks:
        # Ищем <review_text>...</review_text>
        match = re.search(r"<review_text>(.*?)</review_text>", block, re.DOTALL)
        if match:
            review = match.group(1).strip()
        else:
            # Если нет <review_text>, убираем все теги и берём текст
            review = re.sub(r"<.*?>", "", block).strip()
        
        if review:  # пропускаем пустые строки
            reviews.append(review)
    
    return reviews

# Загружаем данные
positive_reviews = extract_reviews(positive_path)
negative_reviews = extract_reviews(negative_path)

print(f"Положительных отзывов: {len(positive_reviews)}")
print(f"Отрицательных отзывов: {len(negative_reviews)}")

import string

# Минимальный набор английских стоп-слов (можно расширить)
stop_words = set([
    "i", "me", "my", "myself", "we", "our", "you", "your", "yours", "he", "him", "his", "she", "her", "it", "its",
    "they", "them", "their", "what", "which", "who", "whom", "this", "that", "am", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but",
    "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on",
    "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all",
    "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same",
    "so", "than", "too", "very", "can", "will", "just"
])

def clean_review(text):
    # Приводим к нижнему регистру
    text = text.lower()
    # Удаляем пунктуацию и цифры
    text = re.sub(r"[^a-z\s]", "", text)
    # Токенизация
    words = text.split()
    # Удаление стоп-слов
    words = [word for word in words if word not in stop_words]
    # Собираем обратно строку
    return " ".join(words)

# Применим очистку
positive_cleaned = list({clean_review(r) for r in positive_reviews})
negative_cleaned = list({clean_review(r) for r in negative_reviews})


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import numpy as np

# Объединяем данные
texts = positive_cleaned + negative_cleaned
labels = [1] * len(positive_cleaned) + [0] * len(negative_cleaned)

# Преобразуем в numpy для совместимости
texts = np.array(texts)
labels = np.array(labels)

# Векторизация текста (TF-IDF)
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(texts)

# Разделение на обучающую, валидационную и тестовую выборки
X_train, X_temp, y_train, y_temp = train_test_split(X, labels, test_size=0.3, random_state=42, stratify=labels)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print("Размеры выборок:")
print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Обучение модели
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Предсказание
y_pred = model.predict(X_test)

# Метрики
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

import joblib

# Путь для сохранения
model_path = r"C:\Users\Admin\Desktop\Intelligent Systems\Assessment 3\NLP-Project\sentiment_model.joblib"
vectorizer_path = r"C:\Users\Admin\Desktop\Intelligent Systems\Assessment 3\NLP-Project\tfidf_vectorizer.joblib"

# Сохранение модели и векторизатора
joblib.dump(model, model_path)
joblib.dump(vectorizer, vectorizer_path)

print("Модель и векторизатор успешно сохранены!")
