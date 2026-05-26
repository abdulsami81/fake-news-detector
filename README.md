# 🔍 Fake News Detector

An NLP-based web app that detects whether a news article is 
Real or Fake with 98%+ accuracy.

## 🔧 Tech Stack
- Python, pandas, numpy
- NLTK, scikit-learn, TF-IDF
- Streamlit (web app)

## 📊 Dataset
Fake and Real News Dataset — Kaggle
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

## 🚀 How to Run

1. Clone the repo
   git clone https://github.com/abdulsami81/fake-news-detector.git

2. Install dependencies
   pip install -r requirements.txt

3. Train the model first
   Run notebooks/02_model.ipynb

4. Run the app
   streamlit run app.py

## 📁 Project Structure
fake-news-detector/
├── data/              ← Fake.csv + True.csv
├── notebooks/         ← EDA + Model training
├── app.py             ← Streamlit web app
├── model.pkl          ← Trained model
├── tfidf.pkl          ← TF-IDF vectorizer
└── README.md

## 📈 Model Results
| Model                | Accuracy |
|----------------------|----------|
| Logistic Regression  | 98.5%    |
| Random Forest        | 99.1%    |