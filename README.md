# 📧 Email Spam Detector

A machine learning project that classifies email messages as **Spam** or **Ham** using NLP and Naive Bayes.

---

## 📁 Project Structure

```
spam_detector_project/
│
├── spam_detector.ipynb   # Full notebook: EDA → Preprocessing → Model Building
├── app.py                # Streamlit web app for live predictions
├── requirements.txt      # Python dependencies
├── spam.csv              # Dataset (add this file before running)
├── model.pkl             # Trained model (generated after running notebook)
├── tfidf.pkl             # TF-IDF vectorizer (generated after running notebook)
└── threshold.pkl         # Tuned threshold (generated after running notebook)
```

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add the dataset
Place `spam.csv` in the project folder. It should have two columns:
- `Category` — `spam` or `ham`
- `Message` — the email text

### 3. Run the notebook
Open and run `spam_detector.ipynb` top to bottom. This will:
- Clean and explore the data
- Train 3 models (Naive Bayes, Logistic Regression, Random Forest)
- Save `model.pkl`, `tfidf.pkl`, and `threshold.pkl`

### 4. Launch the Streamlit app
```bash
streamlit run app.py
```

---

## 🧠 Pipeline Summary

| Step | Description |
|------|-------------|
| Data Cleaning | Remove duplicates, strip whitespace, lowercase |
| EDA | Character/word/sentence distribution, top words |
| Text Preprocessing | Remove stopwords, simple stemming, punctuation removal |
| Vectorization | TF-IDF (3000 features) |
| Models Trained | Naive Bayes, Logistic Regression, Random Forest |
| Best Model | Naive Bayes (Tuned Threshold) |

---

## 📊 Model Results

| Variant | Precision | Recall | Accuracy |
|---------|-----------|--------|----------|
| NB Original | 99.1% | 85.0% | 98% |
| **NB Tuned Threshold ✅** | **97.5%** | **90.8%** | **99%** |
| ComplementNB | 64.0% | 95.0% | 93% |
| NB Bigrams + 5000 | 100.0% | 83.0% | 98% |

---

## 🛠 Tech Stack
- Python, Pandas, Matplotlib, Seaborn
- Scikit-learn (MultinomialNB, TF-IDF, train/test split)
- Streamlit (web app)
- Pickle (model serialization)
