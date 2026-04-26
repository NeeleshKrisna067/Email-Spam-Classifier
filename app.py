import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import Counter

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_auc_score, roc_curve, precision_recall_curve, f1_score,
    precision_score, recall_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# NLP Libraries
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

import warnings
warnings.filterwarnings('ignore')

# ─── NLTK Downloads ────────────────────────────────────────────────────────────
@st.cache_resource
def download_nltk_resources():
    for resource in ['punkt', 'stopwords', 'wordnet', 'omw-1.4', 'punkt_tab']:
        try:
            nltk.download(resource, quiet=True)
        except:
            pass

download_nltk_resources()

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SpamShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* Root Variables */
:root {
    --bg-primary: #0a0e1a;
    --bg-secondary: #111827;
    --bg-card: #1a2035;
    --accent-green: #00ff88;
    --accent-red: #ff3366;
    --accent-yellow: #ffdd44;
    --accent-blue: #4488ff;
    --text-primary: #e8edf5;
    --text-muted: #8892a4;
    --border: #2a3450;
}

/* Global */
.stApp {
    background-color: var(--bg-primary);
    font-family: 'DM Sans', sans-serif;
    color: var(--text-primary);
}

/* Hide default header */
header[data-testid="stHeader"] { background: transparent; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .stMarkdown { color: var(--text-primary); }

/* Custom Hero Banner */
.hero-banner {
    background: linear-gradient(135deg, #0a0e1a 0%, #111827 40%, #0d1f3c 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(0,255,136,0.06) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    color: var(--accent-green);
    letter-spacing: -1px;
    margin: 0 0 0.4rem 0;
}
.hero-subtitle {
    font-size: 1rem;
    color: var(--text-muted);
    font-weight: 300;
    letter-spacing: 0.5px;
}
.hero-badge {
    display: inline-block;
    background: rgba(0,255,136,0.1);
    border: 1px solid rgba(0,255,136,0.3);
    color: var(--accent-green);
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    padding: 3px 10px;
    border-radius: 20px;
    margin-bottom: 1rem;
    letter-spacing: 1px;
}

/* Metric Cards */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: var(--accent-green);
}
.metric-label {
    font-size: 0.78rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
}

/* Result Cards */
.result-spam {
    background: linear-gradient(135deg, rgba(255,51,102,0.12), rgba(255,51,102,0.04));
    border: 2px solid var(--accent-red);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    animation: pulse-red 2s infinite;
}
.result-ham {
    background: linear-gradient(135deg, rgba(0,255,136,0.12), rgba(0,255,136,0.04));
    border: 2px solid var(--accent-green);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    animation: pulse-green 2s infinite;
}
@keyframes pulse-red {
    0%, 100% { box-shadow: 0 0 0 0 rgba(255,51,102,0.2); }
    50% { box-shadow: 0 0 20px 4px rgba(255,51,102,0.15); }
}
@keyframes pulse-green {
    0%, 100% { box-shadow: 0 0 0 0 rgba(0,255,136,0.2); }
    50% { box-shadow: 0 0 20px 4px rgba(0,255,136,0.15); }
}
.result-icon { font-size: 3.5rem; margin-bottom: 0.5rem; }
.result-label {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
}
.result-confidence {
    font-size: 0.9rem;
    color: var(--text-muted);
    margin-top: 0.4rem;
}

/* Section Headers */
.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 1rem;
    color: var(--accent-blue);
    text-transform: uppercase;
    letter-spacing: 2px;
    margin: 1.5rem 0 1rem 0;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* Streamlit overrides */
.stButton > button {
    background: var(--accent-green) !important;
    color: #0a0e1a !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    letter-spacing: 1px !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: #00cc6e !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(0,255,136,0.3) !important;
}

/* Text areas and inputs */
.stTextArea textarea, .stTextInput input {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: var(--accent-green) !important;
    box-shadow: 0 0 0 2px rgba(0,255,136,0.15) !important;
}

/* Selectbox */
.stSelectbox > div > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: transparent;
    border-bottom: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px 8px 0 0 !important;
    color: var(--text-muted) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
    padding: 0.5rem 1rem !important;
}
.stTabs [aria-selected="true"] {
    background: var(--bg-card) !important;
    border-color: var(--accent-green) !important;
    color: var(--accent-green) !important;
}

/* DataFrame */
.stDataFrame { border: 1px solid var(--border); border-radius: 8px; }

/* Info/Warning/Error */
.stAlert { border-radius: 8px !important; border: 1px solid var(--border) !important; }

/* Progress bar */
.stProgress > div > div > div {
    background: var(--accent-green) !important;
}

/* Spinner */
.stSpinner > div { border-top-color: var(--accent-green) !important; }

/* Feature tag */
.feature-tag {
    display: inline-block;
    background: rgba(68,136,255,0.1);
    border: 1px solid rgba(68,136,255,0.3);
    color: var(--accent-blue);
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    padding: 2px 8px;
    border-radius: 4px;
    margin: 2px;
}

/* Divider */
.custom-divider {
    height: 1px;
    background: linear-gradient(to right, transparent, var(--border), transparent);
    margin: 1.5rem 0;
}
</style>
""", unsafe_allow_html=True)


# ─── Text Preprocessing ────────────────────────────────────────────────────────
class EmailPreprocessor:
    def __init__(self, use_stemming=True, use_lemmatization=False, remove_stopwords=True):
        self.use_stemming = use_stemming
        self.use_lemmatization = use_lemmatization
        self.remove_stopwords = remove_stopwords
        self.stemmer = PorterStemmer()
        try:
            self.lemmatizer = WordNetLemmatizer()
        except:
            self.lemmatizer = None
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()

    def extract_features(self, text):
        """Extract hand-crafted features from email text."""
        features = {
            'char_count': len(text),
            'word_count': len(text.split()),
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'dollar_count': text.count('$'),
            'url_count': len(re.findall(r'http[s]?://\S+', text)),
            'digit_ratio': sum(1 for c in text if c.isdigit()) / max(len(text), 1),
            'has_html': int(bool(re.search(r'<[a-zA-Z]', text))),
        }
        return features

    def clean(self, text):
        """Full preprocessing pipeline."""
        if not isinstance(text, str):
            text = str(text)
        # Lowercase
        text = text.lower()
        # Remove URLs
        text = re.sub(r'http[s]?://\S+|www\.\S+', ' urltoken ', text)
        # Remove email addresses
        text = re.sub(r'\S+@\S+', ' emailtoken ', text)
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        # Replace currency symbols
        text = re.sub(r'[$€£¥]', ' moneytoken ', text)
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Tokenize
        try:
            tokens = word_tokenize(text)
        except:
            tokens = text.split()
        # Remove stopwords
        if self.remove_stopwords and self.stop_words:
            tokens = [t for t in tokens if t not in self.stop_words and len(t) > 1]
        # Stemming / Lemmatization
        if self.use_lemmatization and self.lemmatizer:
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        elif self.use_stemming:
            tokens = [self.stemmer.stem(t) for t in tokens]
        return ' '.join(tokens)


# ─── Sample Dataset Generator ─────────────────────────────────────────────────
def generate_sample_dataset():
    spam_emails = [
        "WINNER!! As a valued network customer you have been selected to receive a £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.",
        "Congratulations! You've won a FREE iPhone 15! Click here to claim your prize NOW!! Limited time offer!!!",
        "Get rich quick! Make $5000 a week working from home. No experience needed. Call now!",
        "URGENT: Your bank account has been suspended. Click here immediately to verify your details.",
        "FREE OFFER: Lose 30 pounds in 30 days! Miracle diet pill. Order now and get 50% off!!!",
        "You have been pre-approved for a $10,000 loan. No credit check required. Apply NOW!",
        "Dear friend, I am a Nigerian prince and need your help transferring $15 million...",
        "Enlarge your income! Work from home. Earn $500/day guaranteed. No skills required.",
        "CLICK HERE: Hot singles in your area want to meet you tonight!",
        "Congratulations! Your email was randomly selected. You won $1,000,000! Claim immediately.",
        "BUY NOW! 90% discount on luxury watches. Limited stock. Free shipping worldwide!!!",
        "WARNING: Your computer is infected with viruses! Download our FREE scanner immediately.",
        "Make money fast! Join our MLM network and earn passive income every day.",
        "SPECIAL OFFER: Viagra, Cialis available without prescription. Lowest prices guaranteed!",
        "You have 1 unread message from a secret admirer. Click to reveal who it is!",
        "FREE credit score check! No registration needed. Get yours now before it expires!",
        "Earn extra cash by filling surveys! $50 per survey. Sign up FREE today!",
        "LAST CHANCE: Renew your subscription to avoid account deletion. Act NOW!",
        "Exclusive deal for you only! Investment opportunity 300% ROI guaranteed.",
        "Your PayPal account needs verification. Login immediately to avoid suspension.",
        "Hot deals! Up to 80% off brand name items. Shop now before stock runs out!!!!",
        "You have inherited $2.5 million from a distant relative. Contact us to claim.",
        "FREE trial! Best diet supplement ever! No credit card required. Limited offer!",
        "ALERT: Unusual login detected. Verify your account now or it will be locked.",
        "Double your Bitcoin in 24 hours! Guaranteed returns! Send 0.1 BTC to receive 0.2 BTC",
        "Your Amazon package could not be delivered. Click to reschedule immediately.",
        "Sexy singles are waiting to chat with you! FREE sign up. No obligations.",
        "Win big at our online casino! 100 free spins no deposit required!!! Join now!",
        "BREAKING: New IRS regulation could double your tax refund. Click to find out!",
        "Cheap meds online! No prescription. Overnight shipping. Best prices guaranteed!!!",
    ]

    ham_emails = [
        "Hi John, just wanted to confirm our meeting tomorrow at 2pm. Let me know if the time still works for you.",
        "Please find attached the quarterly financial report for your review. Let me know if you have any questions.",
        "Hey, are you coming to Sarah's birthday party on Saturday? It starts at 7pm at her place.",
        "The project deadline has been extended to next Friday. Please update your timelines accordingly.",
        "Thanks for your email. I'll get back to you by end of business day tomorrow.",
        "Reminder: Team standup call at 9am tomorrow. Agenda: sprint review and planning for next week.",
        "Your order #12345 has been shipped and will arrive within 3-5 business days.",
        "I reviewed the proposal you sent. I have a few suggestions - can we schedule a call to discuss?",
        "Just checking in to see how the onboarding is going. Let me know if you need any help.",
        "The company picnic is scheduled for July 15th. Please RSVP by July 8th if you plan to attend.",
        "Hi, I'm following up on my previous email regarding the contract renewal. Have you had a chance to review it?",
        "Your flight booking confirmation: Flight AA123, departing NYC June 12 at 8:45am. Booking ref: XYZ789.",
        "Good morning! Just a heads up that I'll be working from home today due to a doctor's appointment.",
        "The client presentation went really well! They loved our proposal and are ready to move forward.",
        "Can you please send me the meeting notes from yesterday's strategy session? I missed the last part.",
        "Your monthly bank statement for May is now available. Log in to view your transactions.",
        "Hi Professor Smith, I wanted to ask about the assignment due next week. Is an extension possible?",
        "Dinner plans for Friday? I was thinking we could try that new Italian restaurant downtown.",
        "The IT team has completed the server migration. All systems are back online and running normally.",
        "Happy birthday! Hope you have a wonderful day filled with joy and celebration.",
        "Please review and sign the attached NDA before our meeting next week. Thank you.",
        "The new employee handbook has been updated. Please read through the changes by end of month.",
        "I'm writing to follow up on my job application submitted last week. I'm very interested in the position.",
        "Your appointment with Dr. Johnson is confirmed for Tuesday, June 14th at 10:30am.",
        "The Wi-Fi will be down for maintenance this Sunday from 2am to 5am. Plan accordingly.",
        "Great work on the presentation today! The client was very impressed with the data analysis.",
        "Could you please send me the contact info for the vendor we used last quarter?",
        "Mom, don't forget we're having dinner together on Sunday. I'll pick you up at 6pm.",
        "The library book you requested is now available for pickup. It will be held for 7 days.",
        "Team, please remember to submit your timesheets before the end of day Friday.",
    ]

    emails = spam_emails + ham_emails
    labels = ['spam'] * len(spam_emails) + ['ham'] * len(ham_emails)
    df = pd.DataFrame({'text': emails, 'label': labels})
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


# ─── Model Trainer ─────────────────────────────────────────────────────────────
class SpamClassifier:
    def __init__(self):
        self.models = {}
        self.pipelines = {}
        self.preprocessor = EmailPreprocessor()
        self.vectorizer_type = 'tfidf'
        self.trained = False
        self.results = {}
        self.label_encoder = LabelEncoder()

    def get_vectorizer(self):
        if self.vectorizer_type == 'tfidf':
            return TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2),
                sublinear_tf=True,
                min_df=1,
            )
        else:
            return CountVectorizer(max_features=10000, ngram_range=(1, 2), min_df=1)

    def get_model_zoo(self):
        return {
            'Naive Bayes (Multinomial)': MultinomialNB(alpha=0.1),
            'Naive Bayes (Complement)': ComplementNB(alpha=0.1),
            'Logistic Regression': LogisticRegression(
                C=1.0, max_iter=1000, solver='lbfgs', random_state=42
            ),
            'Linear SVM': LinearSVC(C=1.0, max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            ),
        }

    def preprocess_data(self, df):
        df = df.copy()
        df['cleaned_text'] = df['text'].apply(self.preprocessor.clean)
        return df

    def train(self, df, selected_models, vectorizer_type='tfidf', test_size=0.2):
        self.vectorizer_type = vectorizer_type
        self.trained = False
        self.results = {}

        df = self.preprocess_data(df)
        X = df['cleaned_text']
        y = df['label']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        model_zoo = self.get_model_zoo()
        all_results = {}

        for name in selected_models:
            if name not in model_zoo:
                continue
            model = model_zoo[name]
            vectorizer = self.get_vectorizer()

            pipeline = Pipeline([
                ('vectorizer', vectorizer),
                ('classifier', model)
            ])

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            # Probabilities (not all models support predict_proba)
            try:
                y_proba = pipeline.predict_proba(X_test)[:, 1]
                auc = roc_auc_score((y_test == 'spam').astype(int), y_proba)
            except:
                y_proba = None
                auc = None

            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, pos_label='spam', zero_division=0),
                'recall': recall_score(y_test, y_pred, pos_label='spam', zero_division=0),
                'f1': f1_score(y_test, y_pred, pos_label='spam', zero_division=0),
                'auc': auc,
                'confusion_matrix': confusion_matrix(y_test, y_pred, labels=['spam', 'ham']),
                'report': classification_report(y_test, y_pred, output_dict=True),
                'y_test': y_test,
                'y_pred': y_pred,
                'y_proba': y_proba,
            }

            all_results[name] = metrics
            self.pipelines[name] = pipeline

        self.results = all_results
        self.trained = True
        self.best_model_name = max(all_results, key=lambda k: all_results[k]['f1'])
        return all_results

    def predict(self, text, model_name=None):
        if not self.trained:
            return None, None, None
        if model_name is None:
            model_name = self.best_model_name
        pipeline = self.pipelines.get(model_name)
        if pipeline is None:
            return None, None, None

        cleaned = self.preprocessor.clean(text)
        prediction = pipeline.predict([cleaned])[0]

        try:
            proba = pipeline.predict_proba([cleaned])[0]
            classes = pipeline.classes_
            spam_idx = list(classes).index('spam')
            confidence = proba[spam_idx] if prediction == 'spam' else 1 - proba[spam_idx]
            spam_prob = proba[spam_idx]
        except:
            confidence = 1.0
            spam_prob = 1.0 if prediction == 'spam' else 0.0

        return prediction, confidence, spam_prob

    def get_top_features(self, model_name=None, n=20):
        if not self.trained:
            return [], []
        if model_name is None:
            model_name = self.best_model_name
        pipeline = self.pipelines.get(model_name)
        if not pipeline:
            return [], []
        try:
            vec = pipeline.named_steps['vectorizer']
            clf = pipeline.named_steps['classifier']
            feature_names = vec.get_feature_names_out()

            if hasattr(clf, 'coef_'):
                coefs = clf.coef_
                if coefs.ndim > 1:
                    coefs = coefs[0]
                top_spam = np.argsort(coefs)[-n:][::-1]
                top_ham = np.argsort(coefs)[:n]
                spam_features = [(feature_names[i], coefs[i]) for i in top_spam]
                ham_features = [(feature_names[i], coefs[i]) for i in top_ham]
                return spam_features, ham_features
            elif hasattr(clf, 'feature_log_prob_'):
                log_prob = clf.feature_log_prob_
                classes = list(pipeline.classes_)
                spam_idx = classes.index('spam') if 'spam' in classes else 0
                ham_idx = classes.index('ham') if 'ham' in classes else 1
                spam_log = log_prob[spam_idx]
                ham_log = log_prob[ham_idx]
                diff = spam_log - ham_log
                top_spam_idx = np.argsort(diff)[-n:][::-1]
                top_ham_idx = np.argsort(diff)[:n]
                spam_features = [(feature_names[i], diff[i]) for i in top_spam_idx]
                ham_features = [(feature_names[i], diff[i]) for i in top_ham_idx]
                return spam_features, ham_features
        except Exception as e:
            pass
        return [], []


# ─── Session State ─────────────────────────────────────────────────────────────
if 'classifier' not in st.session_state:
    st.session_state.classifier = SpamClassifier()
if 'training_done' not in st.session_state:
    st.session_state.training_done = False
if 'df' not in st.session_state:
    st.session_state.df = generate_sample_dataset()


# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='font-family: Space Mono, monospace; font-size: 1.2rem; color: #00ff88; 
                font-weight: 700; padding: 1rem 0 0.5rem 0;'>
    🛡️ SpamShield AI
    </div>
    <div style='color: #8892a4; font-size: 0.8rem; margin-bottom: 1.5rem;'>
    Advanced Email Classifier
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ Configuration")

    vectorizer = st.selectbox(
        "Feature Extraction",
        ["tfidf", "count"],
        format_func=lambda x: "TF-IDF Vectorizer" if x == "tfidf" else "Count Vectorizer",
        help="TF-IDF generally performs better for spam classification."
    )

    test_split = st.slider("Test Split Size", 0.1, 0.4, 0.2, 0.05,
                           help="Fraction of data used for testing.")

    st.markdown("### 🤖 Select Models")
    model_options = [
        'Naive Bayes (Multinomial)',
        'Naive Bayes (Complement)',
        'Logistic Regression',
        'Linear SVM',
        'Random Forest',
    ]
    selected_models = []
    for m in model_options:
        if st.checkbox(m, value=(m in ['Naive Bayes (Complement)', 'Logistic Regression', 'Linear SVM']),
                       key=f"model_{m}"):
            selected_models.append(m)

    st.markdown("### 📊 Dataset")
    data_source = st.radio("Data Source", ["Built-in Sample", "Upload CSV"])

    uploaded_df = None
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("CSV with 'text' and 'label' columns", type=['csv'])
        if uploaded_file:
            try:
                uploaded_df = pd.read_csv(uploaded_file)
                if 'text' in uploaded_df.columns and 'label' in uploaded_df.columns:
                    st.success(f"✅ Loaded {len(uploaded_df)} rows")
                else:
                    st.error("CSV must have 'text' and 'label' columns.")
                    uploaded_df = None
            except Exception as e:
                st.error(f"Error: {e}")

    st.markdown("---")
    train_btn = st.button("🚀 TRAIN MODELS", use_container_width=True)
    if not selected_models:
        st.warning("⚠️ Select at least one model")

    st.markdown("""
    <div style='margin-top: 2rem; color: #8892a4; font-size: 0.72rem; line-height: 1.6;'>
    Built with scikit-learn · NLTK · Streamlit<br>
    Supports TF-IDF / BoW features<br>
    5 classifier algorithms
    </div>
    """, unsafe_allow_html=True)


# ─── Training Logic ────────────────────────────────────────────────────────────
if train_btn and selected_models:
    df_to_use = uploaded_df if uploaded_df is not None else st.session_state.df
    with st.spinner("Training models..."):
        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress.progress(i + 1)
        results = st.session_state.classifier.train(
            df_to_use, selected_models, vectorizer, test_split
        )
        st.session_state.training_done = True
        st.session_state.results = results
    st.success(f"✅ {len(results)} model(s) trained successfully!")
    time.sleep(0.5)
    st.rerun()


# ─── Main Content ──────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero-banner'>
  <div class='hero-badge'>🔬 ML-POWERED CLASSIFICATION</div>
  <div class='hero-title'>SpamShield AI</div>
  <div class='hero-subtitle'>
    Intelligent email spam detection using machine learning & NLP.<br>
    Train multiple models, compare performance, and classify emails in real-time.
  </div>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🎯 CLASSIFY", "📊 ANALYTICS", "🔬 DATASET", "📖 FEATURES"])

# ─────────────── TAB 1: CLASSIFY ───────────────────────────────────────────────
with tab1:
    if not st.session_state.training_done:
        st.markdown("""
        <div style='background: rgba(68,136,255,0.08); border: 1px solid rgba(68,136,255,0.3);
                    border-radius: 12px; padding: 2rem; text-align: center; margin: 2rem 0;'>
            <div style='font-size: 3rem; margin-bottom: 1rem;'>🤖</div>
            <div style='font-family: Space Mono, monospace; color: #4488ff; font-size: 1.1rem;
                        margin-bottom: 0.5rem;'>No Models Trained Yet</div>
            <div style='color: #8892a4; font-size: 0.9rem;'>
                Select models in the sidebar and click <strong style='color: #e8edf5;'>TRAIN MODELS</strong> 
                to get started.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        clf = st.session_state.classifier
        col_l, col_r = st.columns([3, 2])

        with col_l:
            st.markdown("<div class='section-header'>📧 Input Email</div>", unsafe_allow_html=True)
            email_text = st.text_area(
                "Paste email content here",
                height=220,
                placeholder="Paste the email subject and body here...\n\nExample:\nCongratulations! You have won a FREE iPhone! Click here to claim your prize NOW!!",
                label_visibility="collapsed"
            )

            model_choice = st.selectbox(
                "Classifier Model",
                list(clf.pipelines.keys()),
                index=list(clf.pipelines.keys()).index(clf.best_model_name)
                if clf.best_model_name in clf.pipelines else 0,
                help="Choose which trained model to use for classification."
            )

            classify_btn = st.button("🔍 ANALYZE EMAIL", use_container_width=True)

        with col_r:
            st.markdown("<div class='section-header'>🎲 Quick Tests</div>", unsafe_allow_html=True)
            spam_examples = [
                "WINNER!! You've been selected to receive £900 prize! Call 09061701461 to claim now! Limited time!!!",
                "Make $5000/week from home! No experience needed. FREE training. Click NOW!",
                "URGENT: Your bank account suspended. Verify immediately or lose access!",
            ]
            ham_examples = [
                "Hi team, reminder about our 3pm meeting today. Please bring your weekly updates.",
                "Your order #45678 has shipped. Expected delivery: Thursday, June 15.",
            ]
            st.markdown("<div style='font-size:0.8rem; color:#8892a4; margin-bottom:0.5rem;'>🚨 Spam Examples</div>", unsafe_allow_html=True)
            for ex in spam_examples:
                if st.button(ex[:55] + "...", key=f"spam_{ex[:20]}", use_container_width=True):
                    st.session_state['example_text'] = ex
            st.markdown("<div style='font-size:0.8rem; color:#8892a4; margin: 0.8rem 0 0.5rem 0;'>✅ Ham Examples</div>", unsafe_allow_html=True)
            for ex in ham_examples:
                if st.button(ex[:55] + "...", key=f"ham_{ex[:20]}", use_container_width=True):
                    st.session_state['example_text'] = ex

        # Auto-fill from example button
        if 'example_text' in st.session_state and not email_text:
            email_text = st.session_state.pop('example_text')
            st.rerun()

        # Classification Result
        if classify_btn and email_text.strip():
            prediction, confidence, spam_prob = clf.predict(email_text.strip(), model_choice)

            st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
            st.markdown("<div class='section-header'>📋 Classification Result</div>", unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-value' style='color:{"#ff3366" if prediction=="spam" else "#00ff88"};'>
                        {"SPAM" if prediction=="spam" else "HAM"}
                    </div>
                    <div class='metric-label'>Verdict</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-value'>{spam_prob:.1%}</div>
                    <div class='metric-label'>Spam Probability</div>
                </div>""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-value'>{1-spam_prob:.1%}</div>
                    <div class='metric-label'>Ham Probability</div>
                </div>""", unsafe_allow_html=True)
            with c4:
                word_count = len(email_text.split())
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-value'>{word_count}</div>
                    <div class='metric-label'>Word Count</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Probability bar chart
            fig, ax = plt.subplots(figsize=(9, 1.5))
            fig.patch.set_facecolor('#1a2035')
            ax.set_facecolor('#1a2035')
            bar_colors = ['#ff3366', '#00ff88']
            bars = ax.barh(['Spam', 'Ham'], [spam_prob, 1-spam_prob],
                           color=bar_colors, height=0.5, edgecolor='none')
            ax.set_xlim(0, 1)
            ax.axvline(0.5, color='#2a3450', linewidth=1.5, linestyle='--')
            for bar, val in zip(bars, [spam_prob, 1-spam_prob]):
                ax.text(min(val + 0.02, 0.95), bar.get_y() + bar.get_height()/2,
                        f'{val:.1%}', va='center', color='white',
                        fontsize=10, fontfamily='monospace')
            ax.tick_params(colors='#8892a4', labelsize=10)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_xlabel('Probability', color='#8892a4', fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Text analysis
            preprocessor = clf.preprocessor
            features = preprocessor.extract_features(email_text)
            st.markdown("<div class='section-header'>🔍 Text Analysis</div>", unsafe_allow_html=True)
            fa, fb, fc = st.columns(3)
            with fa:
                st.metric("Uppercase Ratio", f"{features['uppercase_ratio']:.1%}")
                st.metric("Exclamation Marks", features['exclamation_count'])
            with fb:
                st.metric("$ Dollar Signs", features['dollar_count'])
                st.metric("URLs Found", features['url_count'])
            with fc:
                st.metric("Digit Ratio", f"{features['digit_ratio']:.1%}")
                st.metric("Characters", features['char_count'])

        elif classify_btn:
            st.warning("⚠️ Please enter some email text to classify.")


# ─────────────── TAB 2: ANALYTICS ──────────────────────────────────────────────
with tab2:
    if not st.session_state.training_done:
        st.info("🤖 Train models first to see analytics.")
    else:
        clf = st.session_state.classifier
        results = clf.results

        # Model Comparison Table
        st.markdown("<div class='section-header'>🏆 Model Comparison</div>", unsafe_allow_html=True)
        comparison_data = []
        for name, metrics in results.items():
            comparison_data.append({
                'Model': name,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1']:.4f}",
                'AUC-ROC': f"{metrics['auc']:.4f}" if metrics['auc'] else 'N/A',
            })
        comp_df = pd.DataFrame(comparison_data)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        # Metric Bar Charts
        st.markdown("<div class='section-header'>📈 Performance Metrics</div>", unsafe_allow_html=True)
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        metric_keys = ['accuracy', 'precision', 'recall', 'f1']
        colors = ['#4488ff', '#00ff88', '#ffdd44', '#ff3366']

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        fig.patch.set_facecolor('#111827')
        model_names = list(results.keys())
        short_names = [n.split('(')[0].strip()[:15] for n in model_names]

        for ax, m_name, m_key, color in zip(axes, metric_names, metric_keys, colors):
            ax.set_facecolor('#1a2035')
            values = [results[m][m_key] for m in model_names]
            bars = ax.bar(range(len(model_names)), values, color=color, alpha=0.85,
                          edgecolor='none', width=0.6)
            ax.set_ylim(0, 1.1)
            ax.axhline(1.0, color='#2a3450', linewidth=0.8, linestyle='--')
            ax.set_title(m_name, color='white', fontsize=10, fontweight='bold', pad=8)
            ax.set_xticks(range(len(model_names)))
            ax.set_xticklabels(short_names, rotation=35, ha='right', fontsize=7.5, color='#8892a4')
            ax.tick_params(axis='y', colors='#8892a4', labelsize=8)
            ax.set_ylabel('Score', color='#8892a4', fontsize=8)
            for spine in ax.spines.values():
                spine.set_visible(False)
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=7.5,
                        color='white', fontfamily='monospace')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Confusion Matrices
        st.markdown("<div class='section-header'>🔲 Confusion Matrices</div>", unsafe_allow_html=True)
        n_models = len(results)
        cols_cm = st.columns(min(n_models, 3))
        for i, (name, metrics) in enumerate(results.items()):
            with cols_cm[i % 3]:
                fig, ax = plt.subplots(figsize=(4, 3.5))
                fig.patch.set_facecolor('#1a2035')
                ax.set_facecolor('#1a2035')
                cm = metrics['confusion_matrix']
                cmap = sns.color_palette("dark:#00ff88", as_cmap=True)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=['Spam', 'Ham'],
                            yticklabels=['Spam', 'Ham'],
                            ax=ax, cbar=False,
                            linewidths=1, linecolor='#0a0e1a',
                            annot_kws={'color': 'white', 'fontsize': 14, 'fontweight': 'bold'})
                ax.set_xlabel('Predicted', color='#8892a4', fontsize=9)
                ax.set_ylabel('Actual', color='#8892a4', fontsize=9)
                ax.set_title(name.split('(')[0].strip(), color='white', fontsize=9, fontweight='bold')
                ax.tick_params(colors='#8892a4', labelsize=8)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

        # ROC Curves
        roc_models = {n: m for n, m in results.items() if m['y_proba'] is not None}
        if roc_models:
            st.markdown("<div class='section-header'>📐 ROC Curves</div>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(8, 5))
            fig.patch.set_facecolor('#111827')
            ax.set_facecolor('#1a2035')
            roc_colors = ['#4488ff', '#00ff88', '#ffdd44', '#ff3366', '#ff8833']
            for (name, metrics), color in zip(roc_models.items(), roc_colors):
                y_binary = (metrics['y_test'] == 'spam').astype(int)
                fpr, tpr, _ = roc_curve(y_binary, metrics['y_proba'])
                auc = metrics['auc']
                ax.plot(fpr, tpr, color=color, linewidth=2,
                        label=f"{name.split('(')[0].strip()} (AUC={auc:.3f})")
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1, color='#2a3450', label='Random')
            ax.set_xlabel('False Positive Rate', color='#8892a4')
            ax.set_ylabel('True Positive Rate', color='#8892a4')
            ax.set_title('ROC Curves — All Models', color='white', fontweight='bold')
            ax.legend(loc='lower right', fontsize=8, facecolor='#1a2035',
                      edgecolor='#2a3450', labelcolor='white')
            ax.tick_params(colors='#8892a4')
            for spine in ax.spines.values():
                spine.set_color('#2a3450')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # Top Features
        st.markdown("<div class='section-header'>🔑 Top Discriminating Features</div>", unsafe_allow_html=True)
        feat_model = st.selectbox("Select model for feature analysis",
                                  list(clf.pipelines.keys()), key='feat_model')
        spam_feats, ham_feats = clf.get_top_features(feat_model, n=15)
        if spam_feats:
            fc1, fc2 = st.columns(2)
            with fc1:
                st.markdown("**🚨 Top Spam Indicators**")
                fig, ax = plt.subplots(figsize=(5, 5))
                fig.patch.set_facecolor('#1a2035')
                ax.set_facecolor('#1a2035')
                words = [f[0] for f in spam_feats[:12]]
                scores = [abs(f[1]) for f in spam_feats[:12]]
                ax.barh(words[::-1], scores[::-1], color='#ff3366', alpha=0.8, edgecolor='none')
                ax.tick_params(colors='#8892a4', labelsize=8)
                ax.set_xlabel('Feature Weight', color='#8892a4', fontsize=8)
                for spine in ax.spines.values():
                    spine.set_color('#2a3450')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            with fc2:
                st.markdown("**✅ Top Ham Indicators**")
                fig, ax = plt.subplots(figsize=(5, 5))
                fig.patch.set_facecolor('#1a2035')
                ax.set_facecolor('#1a2035')
                words = [f[0] for f in ham_feats[:12]]
                scores = [abs(f[1]) for f in ham_feats[:12]]
                ax.barh(words[::-1], scores[::-1], color='#00ff88', alpha=0.8, edgecolor='none')
                ax.tick_params(colors='#8892a4', labelsize=8)
                ax.set_xlabel('Feature Weight', color='#8892a4', fontsize=8)
                for spine in ax.spines.values():
                    spine.set_color('#2a3450')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
        else:
            st.info("Feature analysis not available for this model.")


# ─────────────── TAB 3: DATASET ────────────────────────────────────────────────
with tab3:
    df = st.session_state.df
    st.markdown("<div class='section-header'>📋 Dataset Overview</div>", unsafe_allow_html=True)

    d1, d2, d3, d4 = st.columns(4)
    spam_count = (df['label'] == 'spam').sum()
    ham_count = (df['label'] == 'ham').sum()
    with d1:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{len(df)}</div><div class='metric-label'>Total Emails</div></div>", unsafe_allow_html=True)
    with d2:
        st.markdown(f"<div class='metric-card'><div class='metric-value' style='color:#ff3366;'>{spam_count}</div><div class='metric-label'>Spam Emails</div></div>", unsafe_allow_html=True)
    with d3:
        st.markdown(f"<div class='metric-card'><div class='metric-value' style='color:#00ff88;'>{ham_count}</div><div class='metric-label'>Ham Emails</div></div>", unsafe_allow_html=True)
    with d4:
        avg_len = int(df['text'].str.len().mean())
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{avg_len}</div><div class='metric-label'>Avg Length</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    tc1, tc2 = st.columns(2)
    with tc1:
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor('#1a2035')
        ax.set_facecolor('#1a2035')
        wedge_props = {'edgecolor': '#0a0e1a', 'linewidth': 3}
        ax.pie([spam_count, ham_count], labels=['Spam', 'Ham'],
               colors=['#ff3366', '#00ff88'],
               autopct='%1.1f%%', startangle=90,
               textprops={'color': 'white', 'fontsize': 11},
               wedgeprops=wedge_props)
        ax.set_title('Class Distribution', color='white', fontweight='bold', pad=12)
        st.pyplot(fig)
        plt.close()

    with tc2:
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor('#1a2035')
        ax.set_facecolor('#1a2035')
        spam_lens = df[df['label']=='spam']['text'].str.len()
        ham_lens = df[df['label']=='ham']['text'].str.len()
        ax.hist(spam_lens, bins=15, color='#ff3366', alpha=0.7, label='Spam', edgecolor='none')
        ax.hist(ham_lens, bins=15, color='#00ff88', alpha=0.7, label='Ham', edgecolor='none')
        ax.set_xlabel('Email Length (chars)', color='#8892a4')
        ax.set_ylabel('Count', color='#8892a4')
        ax.set_title('Length Distribution', color='white', fontweight='bold')
        ax.legend(facecolor='#1a2035', edgecolor='#2a3450', labelcolor='white', fontsize=9)
        ax.tick_params(colors='#8892a4')
        for spine in ax.spines.values():
            spine.set_color('#2a3450')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("<div class='section-header'>📝 Sample Data</div>", unsafe_allow_html=True)
    filter_label = st.radio("Filter", ["All", "Spam Only", "Ham Only"], horizontal=True)
    filtered_df = df.copy()
    if filter_label == "Spam Only":
        filtered_df = df[df['label'] == 'spam']
    elif filter_label == "Ham Only":
        filtered_df = df[df['label'] == 'ham']

    st.dataframe(
        filtered_df[['label', 'text']].reset_index(drop=True),
        use_container_width=True,
        height=350
    )


# ─────────────── TAB 4: FEATURES / ABOUT ───────────────────────────────────────
with tab4:
    st.markdown("<div class='section-header'>🔬 System Architecture</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background: #1a2035; border: 1px solid #2a3450; border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem;'>
    <h4 style='color: #4488ff; font-family: Space Mono, monospace; margin-bottom: 1rem;'>📦 Pipeline Overview</h4>
    
    <p style='color: #e8edf5;'>The SpamShield AI pipeline processes each email through multiple stages:</p>

    <ol style='color: #8892a4; line-height: 2;'>
        <li><strong style='color: #e8edf5;'>Text Preprocessing</strong> — Lowercasing, URL/email/HTML removal, punctuation stripping, stopword removal</li>
        <li><strong style='color: #e8edf5;'>Tokenization</strong> — NLTK word tokenizer with stemming (Porter) or lemmatization (WordNet)</li>
        <li><strong style='color: #e8edf5;'>Feature Extraction</strong> — TF-IDF or Count Vectorizer with unigrams + bigrams (up to 10k features)</li>
        <li><strong style='color: #e8edf5;'>Classification</strong> — Chosen ML model generates spam/ham prediction with probability</li>
        <li><strong style='color: #e8edf5;'>Post-processing</strong> — Confidence score, hand-crafted feature analysis (uppercase ratio, exclamation count, etc.)</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>🤖 Models Explained</div>", unsafe_allow_html=True)
    models_info = [
        ("Naive Bayes (Multinomial)", "#4488ff", "Classic probabilistic classifier. Works well for discrete word counts. Fast and interpretable. Best for smaller datasets."),
        ("Naive Bayes (Complement)", "#00ff88", "Improved variant that addresses class imbalance by modelling the complement class. Often outperforms standard NB."),
        ("Logistic Regression", "#ffdd44", "Linear discriminative model. Regularized with L2. Excellent F1 and probability calibration. Industry workhorse."),
        ("Linear SVM", "#ff3366", "Finds the optimal separating hyperplane in feature space. No probability output by default. Extremely fast."),
        ("Random Forest", "#ff8833", "Ensemble of 100 decision trees. Captures non-linear patterns. Slower to train but robust to overfitting."),
    ]
    for mname, color, desc in models_info:
        st.markdown(f"""
        <div style='background: #1a2035; border-left: 3px solid {color}; border-radius: 0 10px 10px 0;
                    padding: 0.9rem 1.2rem; margin-bottom: 0.6rem;'>
            <div style='font-family: Space Mono, monospace; color: {color}; font-size: 0.85rem;
                        font-weight: 700; margin-bottom: 0.3rem;'>{mname}</div>
            <div style='color: #8892a4; font-size: 0.85rem; line-height: 1.5;'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>📚 Libraries Used</div>", unsafe_allow_html=True)
    libs = [
        ("scikit-learn", "ML algorithms, pipelines, metrics"),
        ("NLTK", "Tokenization, stopwords, stemming, lemmatization"),
        ("Streamlit", "Interactive web UI"),
        ("TF-IDF / BoW", "Text feature extraction"),
        ("Matplotlib / Seaborn", "Visualizations"),
        ("Pandas / NumPy", "Data manipulation"),
    ]
    lc1, lc2 = st.columns(2)
    for i, (lib, desc) in enumerate(libs):
        with (lc1 if i % 2 == 0 else lc2):
            st.markdown(f"""
            <div style='background: #1a2035; border: 1px solid #2a3450; border-radius: 8px;
                        padding: 0.7rem 1rem; margin-bottom: 0.5rem;'>
                <span style='font-family: Space Mono, monospace; color: #00ff88; font-size: 0.85rem;'>{lib}</span>
                <span style='color: #8892a4; font-size: 0.82rem;'> — {desc}</span>
            </div>
            """, unsafe_allow_html=True)
