from flask import Flask, render_template, request, jsonify
import joblib
from untils import clean_text
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow frontend to call /predict

# ------------------------------
# Load Model + Vectorizer
# ------------------------------
try:
    vec = joblib.load('tfidf_vec.joblib')
    model = joblib.load('fake_news_lr.joblib')
    print("Model and Vectorizer Loaded Successfully!")
except Exception as e:
    print("❌ Error loading model:", e)
    vec = None
    model = None


# ------------------------------
# Home Route
# ------------------------------
@app.route('/')
def index():
    return render_template('index.html')


# ------------------------------
# Prediction API
# ------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if not data or 'text' not in data:
            return jsonify({"error": "No text provided"}), 400

        text = data['text']
        clean = clean_text(text)

        x = vec.transform([clean])
        prob_fake = model.predict_proba(x)[0][1]
        label = "FAKE" if prob_fake > 0.5 else "REAL"

        return jsonify({
            "label": label,
            "confidence": float(round(prob_fake, 3))
        })

    except Exception as e:
        print("❌ Error in /predict:", str(e))
        return jsonify({"error": "Server error", "details": str(e)}), 500


# ------------------------------
# Run Application
# ------------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
