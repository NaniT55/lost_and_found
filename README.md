# 🎒 Campus Lost & Found – Intelligent Matching System  
A machine-learning powered Lost & Found web application that helps match **lost items** with **found items** using:

- TF-IDF text embeddings  
- Optional SVD dimensionality reduction  
- Logistic Regression scoring  
- Jaccard similarity & metadata features  
- Streamlit interactive UI  

This project allows users to describe an item (lost or found) and automatically retrieves the most likely matches from the database of found items.

---

## 🚀 Features

### 🔍 Intelligent Item Matching
- TF-IDF similarity for fast candidate retrieval  
- Logistic Regression model for final ranking  
- Additional handcrafted features:
  - text cosine similarity  
  - title/description/location Jaccard  
  - date difference  
  - color/brand similarity  
  - interaction features  

### 🧠 Machine Learning Pipeline
- Training script (`app.py`) loads train/test CSVs  
- Creates feature vectors & trains logistic regression  
- Saves model artifacts:
  - `tfidf.pkl`
  - `svd.pkl`
  - `scaler.pkl`
  - `logreg_model.pkl`

### 🖥 Streamlit Web App
- Users enter lost item details  
- App retrieves top-K candidate found items  
- Model reranks results based on learned features  
- Users can submit feedback  
- Feedback stored in `streamlit_feedback.csv`  

---


*(Your actual filenames may differ; update as needed.)*

---

## 🧰 Tech Stack

- **Python 3.8+**
- **Streamlit** – UI
- **Scikit-learn** – TF-IDF, SVD, Logistic Regression
- **Pandas / NumPy** – data processing
- Optional: **filelock / SQLite / Postgres** for feedback persistence

---






