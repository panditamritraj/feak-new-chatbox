from flask import Flask, render_template, request, jsonify
import joblib
from untils import clean_text

app = Flask(__name__)

vec = joblib.load('tfidf_vec.joblib')
model = joblib.load('fake_news_lr.joblib')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.json.get('text', '')
    clean = clean_text(text)
    x = vec.transform([clean])
    prob_fake = model.predict_proba(x)[0][1]
    label = "FAKE" if prob_fake > 0.5 else "REAL"
    return jsonify({"label": label, "confidence": round(prob_fake, 3)})

if __name__ == '__main__':
    app.run(debug=True)
