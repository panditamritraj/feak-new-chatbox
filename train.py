import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
from untils import clean_text

DATA_PATH = 'data/toy_news.csv'
VEC_PATH = 'tfidf_vec.joblib'
MODEL_PATH = 'fake_news_lr.joblib'

df = pd.read_csv(DATA_PATH)
df['clean'] = df['text'].apply(clean_text)
df['label_num'] = df['label'].apply(lambda x: 1 if x.upper()=='FAKE' else 0)

X_train, X_test, y_train, y_test = train_test_split(
    df['clean'], df['label_num'], test_size=0.3, random_state=42
)

vec = TfidfVectorizer(max_features=5000)
Xtr = vec.fit_transform(X_train)
Xte = vec.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(Xtr, y_train)

pred = model.predict(Xte)
print(classification_report(y_test, pred))

joblib.dump(vec, VEC_PATH)
joblib.dump(model, MODEL_PATH)
print("Model saved!")
