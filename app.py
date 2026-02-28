import streamlit as st
import numpy as np
import pickle
import time

# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# ===================== REMOVE EXTRA TOP SPACE =====================
st.markdown("""
<style>
.block-container {
    padding-top: 0.8rem;
}
</style>
""", unsafe_allow_html=True)

# ===================== LOAD MODEL & SCALER =====================
model = pickle.load(open("heart_model.pkl", "rb"))
scaler = pickle.load(open("heart_scaler.pkl", "rb"))

# ===================== TITLE =====================
st.markdown(
    "<h1 style='text-align:center; margin-top:0;'>❤️ Heart Disease Prediction System</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;'>Machine Learning based cardiac risk assessment</p>",
    unsafe_allow_html=True
)

st.divider()

# ===================== SIDEBAR INPUTS =====================
st.sidebar.header("🧾 Patient Information")

age = st.sidebar.slider("Age", 20, 100, 45)

sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
sex = 1 if sex == "Male" else 0

cp = st.sidebar.slider("Chest Pain Type (0–3)", 0, 3, 1)

trestbps = st.sidebar.slider("Resting Blood Pressure", 80, 200, 120)

chol = st.sidebar.slider("Cholesterol (mg/dl)", 100, 600, 200)

fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
fbs = 1 if fbs == "Yes" else 0

restecg = st.sidebar.slider("Rest ECG (0–2)", 0, 2, 1)

thalach = st.sidebar.slider("Maximum Heart Rate", 60, 220, 150)

exang = st.sidebar.selectbox("Exercise Induced Angina", ["No", "Yes"])
exang = 1 if exang == "Yes" else 0

oldpeak = st.sidebar.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)

slope = st.sidebar.slider("Slope (0–2)", 0, 2, 1)

ca = st.sidebar.slider("Major Vessels (0–3)", 0, 3, 0)

thal = st.sidebar.slider("Thal (0–3)", 0, 3, 1)

# ===================== MAIN CONTENT =====================
col1, col2 = st.columns(2)

with col1:
    st.subheader("📌 About This App")
    st.write("""
    - Predicts **heart disease risk**
    - Uses trained **Machine Learning model**
    - Helps in early medical awareness
    """)

with col2:
    st.subheader("📊 Input Summary")
    st.json({
        "Age": age,
        "Sex": sex,
        "Chest Pain": cp,
        "Blood Pressure": trestbps,
        "Cholesterol": chol,
        "Max Heart Rate": thalach
    })

st.divider()

# ===================== PREDICTION =====================
if st.button("🔍 Predict Heart Disease Risk", use_container_width=True):

    with st.spinner("Analyzing heart health..."):
        time.sleep(1.5)

        input_data = np.array([
            age, sex, cp, trestbps, chol,
            fbs, restecg, thalach, exang,
            oldpeak, slope, ca, thal
        ]).reshape(1, -1)

        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Heart Disease Detected")
        st.warning("💡 Please consult a cardiologist and maintain a healthy lifestyle.")
    else:
        st.success("✅ Low Risk of Heart Disease")
        st.info("💡 Keep exercising, eat healthy, and monitor regularly.")

# ===================== FOOTER =====================
st.divider()
st.markdown(
    "<p style='text-align:center; color:gray;'>Made with ❤️ using Streamlit & Machine Learning</p>",
    unsafe_allow_html=True
)
