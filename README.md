# 🛡️ SpamShield AI — Email Spam Classifier

An advanced ML-powered email spam classifier built with Python, scikit-learn, NLTK, and Streamlit.

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 🔬 Features

### 🤖 5 ML Models
| Model | Notes |
|-------|-------|
| Naive Bayes (Multinomial) | Classic probabilistic; fast + interpretable |
| Naive Bayes (Complement) | Better for imbalanced classes |
| Logistic Regression | L2 regularized; excellent F1 + probability calibration |
| Linear SVM | Max-margin classifier; extremely fast |
| Random Forest | 100-tree ensemble; robust, non-linear |

### 📊 Analytics Dashboard
- Model comparison table (Accuracy, Precision, Recall, F1, AUC-ROC)
- Confusion matrices for every trained model
- ROC curves overlay
- Top spam/ham discriminating features (bar charts)

### 🔍 Real-time Classification
- Paste any email and classify instantly
- Choose which trained model to use
- Shows spam probability, confidence, and hand-crafted feature signals

### 📁 Dataset Support
- Built-in 60-sample balanced dataset (30 spam + 30 ham)
- Upload your own CSV with `text` and `label` columns
- Popular datasets: [SMS Spam Collection (UCI)](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)

---

## 🧹 NLP Pipeline

```
Raw Email
    → Lowercase
    → Remove URLs / Emails / HTML tags
    → Replace currency symbols → "moneytoken"
    → Strip punctuation
    → NLTK Tokenization
    → Remove stopwords
    → Porter Stemming (or WordNet Lemmatization)
    → TF-IDF / Count Vectorizer (1-2 grams, up to 10k features)
    → ML Classifier
    → Prediction + Probability
```

---

## 📂 Project Structure

```
spam_classifier/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## 🗃️ Using Your Own Dataset

Upload a CSV with these exact column names:
```
text,label
"Hello, your account needs verification","spam"
"Hi Sarah, see you at 3pm tomorrow","ham"
```

Supported label values: `spam` / `ham`

Popular public datasets:
- **SMS Spam Collection** — 5,574 messages (UCI ML Repository)
- **Enron Email Dataset** — Large corporate email dataset
- **SpamAssassin Public Corpus** — Labeled email corpus

---

## ⚙️ Configuration Options

| Setting | Options | Default |
|---------|---------|---------|
| Feature Extraction | TF-IDF, Count Vectorizer | TF-IDF |
| Test Split | 10% – 40% | 20% |
| N-gram Range | (1,2) unigrams + bigrams | Fixed |
| Max Features | 10,000 | Fixed |

---

## 📦 Dependencies

- `streamlit` — Web UI
- `scikit-learn` — ML models, TF-IDF, metrics
- `nltk` — Tokenization, stopwords, stemming
- `pandas` / `numpy` — Data handling
- `matplotlib` / `seaborn` — Visualization
