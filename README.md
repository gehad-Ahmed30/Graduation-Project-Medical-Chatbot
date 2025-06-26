# 🩺 AI Medical Chatbot – Graduation Project

An AI-powered medical chatbot that classifies user input as **medical** or **non-medical**, predicts the most likely **disease**, and suggests the appropriate **medical specialty** based on symptoms. The system uses natural language processing and machine learning to provide intelligent health assistance.

---

## 📌 Project Overview

We developed two supervised machine learning models:

1. **Text Classification Model**
   - **Goal:** Distinguish between medical and non-medical inputs.
   - **Algorithm:** Naive Bayes (Best Accuracy: 99%)
   - **Use:** Filters out unrelated text (sports, tech, etc.)

2. **Disease Prediction Model**
   - **Goal:** Predict diseases based on symptoms and classify them under specialties.
   - **Algorithm:** Support Vector Machine (SVM)
   - **Accuracy:** 
     - Validation Accuracy: **99.43%**
     - Test Accuracy: **99.18%**

---

## 📂 Datasets Used

### 1. Medical Dataset  
- Contains symptom descriptions and disease names.  
- **Medical specialties were manually added.**  
- **Used for disease prediction.**  
- Source: [Kaggle - Symptom to Disease](https://www.kaggle.com/datasets/niyarrbarman/symptom2disease/data)

### 2. Non-Medical Dataset  
- Includes general text in various domains (sports, engineering, etc.)  
- **Used to detect and filter non-medical input.**  
- Source: [Kaggle - News Category Dataset](https://www.kaggle.com/datasets/rmisra/news-category-data)

---

## 🧠 Machine Learning Workflow

1. **Text Cleaning and Preprocessing**  
2. **Tokenization and Vectorization using TF-IDF**  
3. **Model Training and Hyperparameter Tuning using GridSearchCV**
4. **Performance Evaluation using k-Fold Cross Validation**

---

## 💡 Why SVM for Disease Prediction?

- Achieved the highest accuracy
- Efficient for small/medium datasets
- Robust against overfitting
- Excellent with high-dimensional TF-IDF vectors
- Easy to tune using GridSearch

---

## ✅ Overfitting Prevention

- Used **k-Fold Cross Validation**
- Very small gap between validation and test accuracy (**0.25%**)
- **Balanced sampling** of medical/non-medical inputs
- **GridSearchCV** used for reliable hyperparameter tuning

---

## 🤖 Chatbot Behavior

### 🔹 Non-Medical Input  
The model detects unrelated inputs and politely requests the user to describe their symptoms.

### 🔹 Medical Input  
The chatbot detects symptoms, predicts the most likely disease, and suggests the corresponding medical specialty.

### 🔹 Mixed Input  
The model filters out irrelevant parts, focuses on symptoms, and accurately predicts the medical condition.

---

## 📌 Technologies Used

- Python
- Scikit-learn
- Pandas, NumPy
- NLTK
- Joblib
- TF-IDF Vectorizer
- Jupyter Notebook

---

## 📎 Project Structure
``` 
AI-Medical-Chatbot/
│
├── data/
│ ├── medical_dataset.csv
│ └── non_medical_dataset.csv
│
├── models/
│ ├── svm_disease_model.joblib
│ └── naive_bayes_classifier.joblib
│
├── notebooks/
│ └── training_experiments.ipynb
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
``` 
