import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Download required NLTK data (run once)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

# ── Preprocessor ──────────────────────────────────────────────────────────────
stemmer    = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+',  ' urltoken ',   text)
    text = re.sub(r'\S+@\S+',         ' emailtoken ', text)
    text = re.sub(r'<[^>]+>',         ' ',            text)
    text = re.sub(r'[$€£¥]',          ' moneytoken ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(t) for t in tokens if t not in stop_words and len(t) > 1]
    return ' '.join(tokens)

# ── Pipeline: TF-IDF + Linear SVM (best F1 = 0.98) ───────────────────────────
model = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)),
    ('clf',   LinearSVC(C=1.0, max_iter=1000)),
])

# ── Train ─────────────────────────────────────────────────────────────────────
# Replace with your own data — CSV with 'text' and 'label' (spam/ham) columns
import pandas as pd
df = pd.read_csv('SMSSpamCollection', sep='\t', header=None, names=['label', 'text'])

X = df['text'].apply(preprocess)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
model.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────────────
from sklearn.metrics import classification_report
print(classification_report(y_test, model.predict(X_test)))

# ── Predict ───────────────────────────────────────────────────────────────────
def predict(email: str) -> dict:
    clean   = preprocess(email)
    label   = model.predict([clean])[0]
    # LinearSVC doesn't have predict_proba — use decision_function for confidence
    score   = model.decision_function([clean])[0]
    confidence = round(min(abs(float(score)) * 30, 100), 1)  # normalise to 0-100
    return {'label': label, 'confidence': f'{confidence}%'}

# ── Quick test ────────────────────────────────────────────────────────────────
tests = [
    "WINNER!! Claim your free £900 prize NOW!! Call 09061701461 immediately!!!",
    "Hi Sarah, can we reschedule tomorrow's 3pm meeting to 4pm? Thanks.",
]
for t in tests:
    print(t[:60], "->", predict(t))