# 🎯 Sentiment Analysis of Product Reviews

A machine learning web application that classifies product reviews as **Positive**, **Negative**, or **Neutral** using two models — **SVM + TF-IDF** and **DistilBERT** — with an interactive UI built in **Streamlit**.

> **Presented by:** Sebin George · Lubaba Sadiya NP · Manasa Muraleedharan E

---

## 🚀 Live Demo

👉 [https://sentimental-analysis-of-appuct-review.streamlit.app/](https://sentimental-analysis-of-appuct-review.streamlit.app/)

---

## 📌 Features

- 🔍 Classifies reviews into **Positive**, **Negative**, or **Neutral**
- 🤖 Dual-model inference — switch between **DistilBERT** and **SVM** at runtime
- 📊 Confidence bar showing prediction certainty
- ⚠️ Token warning when input exceeds DistilBERT's 128-token limit
- 🎨 Custom dark-themed UI with DM Serif Display font
- ⚡ Fast repeated inference using `@st.cache_resource`

---

## 🗂️ Dataset

- **Source:** Yelp Review Dataset (via HuggingFace `datasets` library)
- **Size:** 10,000 samples
- **Features:** `text`, `label`, `stars`, `sentiment`
- **Label Mapping:**
  - Stars 1–2 → **Negative**
  - Star 3 → **Neutral**
  - Stars 4–5 → **Positive**
- No missing values; clean and ready for training

---

## 🧠 Models

### 1. SVM + TF-IDF
- TF-IDF converts text into numerical vectors
- SVM classifier predicts sentiment class
- **Accuracy: 75.6%**
- Strong performance on Positive and Negative; Neutral class is harder

### 2. DistilBERT (Fine-tuned)
- Pretrained transformer model fine-tuned on the Yelp dataset
- Captures contextual meaning of words
- **Accuracy: 76.3%**
- Higher F1 Score than SVM

---

## 📊 Model Performance

| Model        | Accuracy | F1 Score (macro) |
|--------------|----------|------------------|
| SVM + TF-IDF | 75.6%    | 69.8%            |
| DistilBERT   | 76.3%    | 71.0%            |

### Performance Trend (Line Plot)
![SVM vs DistilBERT Line Plot](lineplot.png)

### Side-by-Side Comparison (Bar Plot)
![SVM vs DistilBERT Bar Plot](barplot.png)

---

## 🛠️ Tech Stack

| Component     | Technology                             |
|---------------|----------------------------------------|
| Language      | Python                                 |
| ML Models     | Scikit-learn, HuggingFace Transformers |
| Deep Learning | PyTorch                                |
| NLP           | TF-IDF, DistilBERT                     |
| Web App       | Streamlit                              |
| Dataset       | HuggingFace `datasets`                 |

---

## 📁 Project Structure
---

## ⚙️ Installation & Running Locally

### 1. Clone the repository
```bash
git clone https://github.com/lubabasadiyanp/sentimental-analysis-of-product-review.git
cd sentimental-analysis-of-product-review
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

---

## 📦 Requirements
---

## ☁️ Deployment (Streamlit Cloud)

1. Push all files including `app.py`, `requirements.txt`, and model files to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in
3. Click **New App** → select your repo, branch, and `app.py` as entry point
4. Click **Deploy** — Streamlit installs packages and launches the app
5. Share the public `*.streamlit.app` URL — no server management needed

---

## 📷 App Interface

![App Interface](app%20interface.jpg)

---

## 👩‍💻 Authors

| Name | Role |
|------|------|
| Sebin George | Streamlit Deployment & App UI |
| Lubaba Sadiya NP | Model Training & Evaluation |
| Manasa Muraleedharan E | Data Preparation & Analysis |

---

## 📄 License

This project is for academic purposes.
