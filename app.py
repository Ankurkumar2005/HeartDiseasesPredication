import streamlit as st
import pandas as pd
import joblib

# Load saved files
model = joblib.load("KNN_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

# App Title
st.title("❤️ Heart Disease Prediction App")

st.markdown("Provide the following details")

# ---- Input Fields ----
age = st.number_input("Age", min_value=1, max_value=120)
sex = st.selectbox("Sex (1 = Male, 0 = Female)", [0, 1])
cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure")
chol = st.number_input("Cholesterol")
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Resting ECG", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved")
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("ST Depression")
slope = st.selectbox("Slope", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia", [0, 1, 2, 3])

# ---- Predict Button ----
if st.button("Predict"):
    input_data = pd.DataFrame([{
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }])

    # Match training columns
    input_data = input_data.reindex(columns=expected_columns, fill_value=0)

    # Scale data
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.error("⚠️ Heart Disease Detected")
    else:
        st.success("✅ No Heart Disease Detected")
