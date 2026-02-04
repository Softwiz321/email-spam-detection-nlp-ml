# Email Spam Detection using NLP and Machine Learning

## Overview
This project builds an end-to-end **spam email classifier** using Natural Language Processing (NLP) and supervised machine learning techniques. The notebook demonstrates how to preprocess raw email text, convert it into numerical features using vectorization, and train classification models to distinguish between **spam** and **ham (non-spam)** emails.

---

## Objective
- Clean and preprocess raw email text data
- Transform text into numerical features using NLP techniques
- Train machine learning classifiers for spam detection
- Evaluate model performance using standard classification metrics

---

## Dataset Description
- **Dataset:** SMS/Email spam dataset (CSV format)
- **Features:**
  - Message text
  - Label (spam / ham)
- **Target Variable:** Spam label (binary classification)

---

## Tools & Technologies
- **Language:** Python  
- **Libraries:**
  - Pandas
  - NumPy
  - Matplotlib
  - Seaborn
  - NLTK
  - Scikit-learn
  - Counter
- **Environment:** Jupyter Notebook

---

## Project Workflow

### 1. Data Loading & Inspection
- Loaded dataset and reviewed class distribution
- Renamed columns for clarity
- Dropped unnecessary columns

---

### 2. Text Preprocessing (NLP)
- Converted text to lowercase
- Removed punctuation and special characters
- Tokenized text
- Removed stopwords
- Applied stemming
- Created a cleaned text column

---

### 3. Feature Extraction
- Converted text into numerical vectors using:
  - **CountVectorizer**
  - **TF-IDF Vectorizer**
- Compared representations for model performance

---

### 4. Model Development
Trained and evaluated multiple classifiers:
- Naive Bayes (MultinomialNB)
- Logistic Regression

---

### 5. Model Evaluation
- Accuracy Score
- Confusion Matrix
- Precision, Recall, F1-score
- Compared model performance across vectorization methods

---

## Key Results
- TF-IDF features performed better than raw count vectors
- Naive Bayes provided strong baseline performance
- Linear SVM / Logistic Regression achieved higher precision and recall
- The final model effectively distinguishes spam from ham messages
