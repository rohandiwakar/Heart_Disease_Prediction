# ❤️ Heart Disease Prediction System

A Machine Learning based web application that predicts the risk of heart disease using patient medical data.

This project uses a trained ML classification model and is deployed using Streamlit for real-time prediction.

---

## 📌 Project Overview

Heart disease is one of the leading causes of death worldwide. Early detection can significantly reduce risks and improve treatment outcomes.

This application:
- Takes patient medical inputs
- Scales the data using a trained scaler
- Uses a pre-trained ML model
- Predicts whether the person is at High Risk or Low Risk of Heart Disease

---

## 🧠 Machine Learning Workflow

1. Data Collection (Heart Dataset - heart.csv)
2. Data Preprocessing & Feature Scaling
3. Model Training
4. Model Saving using Pickle
5. Web App Deployment using Streamlit

Saved Files:
- `heart_model.pkl` → Trained ML Model
- `heart_scaler.pkl` → Feature Scaler
- `app.py` → Streamlit Application
- `heart.csv` → Dataset

---

## 📊 Input Features Used

The model uses the following medical parameters:

- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- Rest ECG
- Maximum Heart Rate
- Exercise Induced Angina
- Oldpeak (ST Depression)
- Slope
- Major Vessels
- Thal

---

## 🖥️ Tech Stack

- Python
- NumPy
- Pandas
- Scikit-Learn
- Pickle
- Streamlit

---

## ⚙️ How to Run This Project

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/heart-disease-prediction.git
```

### 2️⃣ Go to Project Directory
```bash
cd crop-recommendation-system
```

### 3️⃣ Install Required Libraries
```bash
pip install numpy pandas matplotlib scikit-learn
```

### 4️⃣ Run the Project
```bash
streamlit run app.py
```

## 🛠️ Tools Used
```bash
  PyCharm (IDE)
```
