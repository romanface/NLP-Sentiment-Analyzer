#  NLP Sentiment Analyzer – Book Reviews

This project is an individual AI system developed as part of the *Intelligent Systems* course (ISY503 Assessment 3).  
It performs **sentiment analysis** of English-language book reviews from Amazon, classifying them as **positive** or **negative** using Natural Language Processing and Machine Learning techniques.

# Features

- Preprocessing of Amazon `.review` files
- Text cleaning, tokenization, stopword removal
- TF-IDF vectorization
- Logistic Regression classifier (scikit-learn)
- Web interface built with Flask
- Real-time classification of user input

#How it works

1. **Data Preprocessing**  
   - Reviews are extracted from `.review` files  
   - Cleaned from punctuation, numbers, stopwords  
   - Vectorized using TF-IDF (max 5000 features)

2. **Model Training**  
   - Data is split: 70% training, 15% validation, 15% testing  
   - Logistic Regression model trained using scikit-learn  
   - Model achieves ~90% test accuracy  
   - Saved via `joblib` for deployment

3. **Web Interface**  
   - User enters a review in a browser form  
   - The system returns: `Positive` or `Negative`  
   - Built using Flask + HTML + CSS
  
#How to Run

1. Install required packages:
```bash
pip install -r requirements.txt
2. Train the model (optional, already done): python train_and_save_model.py
3. Start the web server: python app.py
4. Open browser: http://127.0.0.1:5000/

Technologies Used
Python 3
scikit-learn
Flask
NumPy
Regular Expressions
TF-IDF vectorization
HTML/CSS (frontend)
