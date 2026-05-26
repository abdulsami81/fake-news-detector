import streamlit as st
import pickle
import re
import string
import nltk

nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

# Models load karo
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

# Page config
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="centered"
)

# Styling
st.markdown("""
    <style>
    .big-title {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        color: #1D9E75;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #888;
        margin-bottom: 2rem;
        font-size: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-title">🔍 Fake News Detector</p>',
            unsafe_allow_html=True)
st.markdown('<p class="subtitle">Paste any news article or headline '
            'to check if it\'s Real or Fake</p>',
            unsafe_allow_html=True)

# Text cleaning
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(
        str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [w for w in tokens
              if w not in stop_words and len(w) > 2]
    return " ".join(tokens)

# Input
news_input = st.text_area(
    "📰 Paste News Article or Headline Here:",
    height=200,
    placeholder="e.g. Scientists discover new vaccine that cures cancer..."
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_btn = st.button("🔍 Analyze News", use_container_width=True)

# Prediction
if predict_btn:
    if not news_input.strip():
        st.warning("⚠️ Please enter some news text first!")
    else:
        with st.spinner("Analyzing..."):
            cleaned   = clean_text(news_input)
            vectorized = tfidf.transform([cleaned])
            prediction = model.predict(vectorized)[0]
            proba      = model.predict_proba(vectorized)[0]
            confidence = max(proba) * 100

        st.markdown("---")

        if prediction == 1:
            st.success(f"✅ REAL NEWS")
            st.metric("Confidence", f"{confidence:.1f}%")
            st.info("This article appears to be from a legitimate source.")
        else:
            st.error(f"🚨 FAKE NEWS")
            st.metric("Confidence", f"{confidence:.1f}%")
            st.warning("This article shows signs of misinformation.")

        # Probability bars
        st.markdown("### Prediction Breakdown")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🚨 Fake Probability", f"{proba[0]*100:.1f}%")
            st.progress(float(proba[0]))
        with col2:
            st.metric("✅ Real Probability", f"{proba[1]*100:.1f}%")
            st.progress(float(proba[1]))

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#888; font-size:12px'>"
    "Built with Python, NLP & Streamlit</p>",
    unsafe_allow_html=True
)