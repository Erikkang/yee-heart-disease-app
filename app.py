import streamlit as st
import joblib
import numpy as np

# Load model and columns
model = joblib.load("heart_model.pkl")
columns = joblib.load("columns.pkl")

st.title("❤️ Heart Disease Risk Predictor")

user_input = {}

# Health-related inputs
st.subheader("📊 Health Metrics")
user_input["BMI"] = st.number_input("BMI", min_value=10.0, step=0.1)
physical_map = {
    "✅ Great (0–3 bad days)": 2,
    "⚖️ Average (4–10 bad days)": 7,
    "❌ Poor (11–30 bad days)": 20
}
user_input["PhysicalHealth"] = physical_map[st.selectbox("Physical Health", list(physical_map.keys()))]
user_input["MentalHealth"] = physical_map[st.selectbox("Mental Health", list(physical_map.keys()))]
sleep_map = {
    "💤 Too little (<4 hrs)": 3,
    "😴 Not enough (4–6 hrs)": 5,
    "✅ Healthy (7–9 hrs)": 8,
    "🛌 Too much (10+ hrs)": 10
}
user_input["SleepTime"] = sleep_map[st.selectbox("Sleep Time", list(sleep_map.keys()))]

# Checkboxes for yes/no fields
st.subheader("✅ Lifestyle & Conditions")
checkbox_fields = [
    "Smoking", "AlcoholDrinking", "Stroke", "DiffWalking",
    "Diabetic", "PhysicalActivity", "Asthma", "KidneyDisease", "SkinCancer"
]
cols = st.columns(3)
for i, field in enumerate(checkbox_fields):
    user_input[field] = 1 if cols[i % 3].checkbox(field) else 0

# Encoded inputs
st.subheader("🧬 Demographics")
user_input["Sex"] = 1 if st.radio("Sex", ["Female", "Male"]) == "Male" else 0
user_input["AgeCategory"] = st.number_input("Age Category (0-13)", 0, 13)
user_input["Race"] = st.number_input("Race (0-5)", 0, 5)
user_input["GenHealth"] = st.number_input("General Health (0=Excellent → 4=Poor)", 0, 4)

# Predict
if st.button("Predict"):
    input_array = np.array([[user_input[col] for col in columns]])
    prediction = model.predict(input_array)[0]
    confidence = model.predict_proba(input_array)[0][1]

    if prediction == 1:
        st.error("🚨 At Risk of Heart Disease!")
    else:
        st.success("✅ Not at Risk.")

    st.info(f"Confidence Score: {confidence:.2f}")
