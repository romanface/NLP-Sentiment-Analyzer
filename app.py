from flask import Flask, request, render_template
import joblib

# Loading the model and vectorizer
model = joblib.load("sentiment_model.joblib")
vectorizer = joblib.load("tfidf_vectorizer.joblib")

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        input_text = request.form["review"]
        transformed = vectorizer.transform([input_text])
        prediction = model.predict(transformed)
        result = "Positive review 😊" if prediction[0] == 1 else "Negative review 😠"
        return render_template("index.html", review=input_text, result=result)

if __name__ == "__main__":
    app.run(debug=True)
