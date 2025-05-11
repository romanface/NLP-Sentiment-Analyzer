import os
import re
import string
import numpy as np
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Set the path to the review files
positive_path = r"C:\Users\Admin\Desktop\Intelligent Systems\Assessment 3\NLP-Project\data\positive.review"
negative_path = r"C:\Users\Admin\Desktop\Intelligent Systems\Assessment 3\NLP-Project\data\negative.review"

def extract_reviews(file_path):
    """Extracts review texts from a .review file"""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    
    # Find <review>...</review> blocks
    review_blocks = re.findall(r"<review>(.*?)</review>", content, re.DOTALL)
    
    reviews = []
    for block in review_blocks:
        # Try to extract <review_text>...</review_text>
        match = re.search(r"<review_text>(.*?)</review_text>", block, re.DOTALL)
        if match:
            review = match.group(1).strip()
        else:
            # If <review_text> is missing, remove tags and use all text
            review = re.sub(r"<.*?>", "", block).strip()
        
        if review:
            reviews.append(review)
    
    return reviews

# Load positive and negative reviews
positive_reviews = extract_reviews(positive_path)
negative_reviews = extract_reviews(negative_path)

print(f"Number of positive reviews: {len(positive_reviews)}")
print(f"Number of negative reviews: {len(negative_reviews)}")

# Define a basic list of English stopwords
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
    """Cleans and preprocesses a single review"""
    text = text.lower()                          # Convert to lowercase
    text = re.sub(r"[^a-z\s]", "", text)         # Remove punctuation and digits
    words = text.split()                         # Tokenize
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    return " ".join(words)

# Apply cleaning
positive_cleaned = list({clean_review(r) for r in positive_reviews})
negative_cleaned = list({clean_review(r) for r in negative_reviews})

# Combine texts and create labels
texts = positive_cleaned + negative_cleaned
labels = [1] * len(positive_cleaned) + [0] * len(negative_cleaned)

# Convert to NumPy arrays
texts = np.array(texts)
labels = np.array(labels)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(texts)

# Split into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, labels, test_size=0.3, random_state=42, stratify=labels)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print("Dataset sizes:")
print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)

# Train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Print evaluation metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Paths to save model and vectorizer
model_path = r"C:\Users\Admin\Desktop\Intelligent Systems\Assessment 3\NLP-Project\sentiment_model.joblib"
vectorizer_path = r"C:\Users\Admin\Desktop\Intelligent Systems\Assessment 3\NLP-Project\tfidf_vectorizer.joblib"

# Save the model and vectorizer
joblib.dump(model, model_path)
joblib.dump(vectorizer, vectorizer_path)

print("Model and vectorizer saved successfully!")

