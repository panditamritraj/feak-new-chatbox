import re
import nltk
from nltk.corpus import stopwords

try:
    STOP = set(stopwords.words('english'))
except:
    nltk.download('stopwords')
    STOP = set(stopwords.words('english'))

def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = [w for w in text.split() if w not in STOP and len(w) > 2]
    return ' '.join(tokens)
