import streamlit as st
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt

# ----------- Load Model and Metadata -----------
@st.cache_resource
def load_resources():
    model = joblib.load("heart_model.pkl")
    with open("metadata.json") as f:
        meta = json.load(f)
    return model, meta

model, meta = load_resources()
numerical = meta["numerical"]
categorical = meta["categorical"]

# ----------- Page Setup -----------
st.set_page_config(page_title="Heart Risk Checker", layout="wide")

st.title("💓 Heart Disease Risk Checker")

# 👉 Instructions directly under the title
st.markdown("""
This tool helps assess whether a patient is at risk of developing heart disease based on lifestyle and clinical information.

### 📌 Instructions:
- Fill in the patient's details in the form below.
- Adjust the risk threshold if needed (default is 0.5).
- Click **Run Prediction** to view the result and confidence score.
""")

# Risk threshold slider
threshold = st.slider("🔧 Set Risk Threshold", 0.0, 1.0, 0.5, 0.01)

# ----------- Input Form -----------
with st.form("risk_form"):
    st.subheader("Patient Clinical Information")

    col1, col2 = st.columns(2)

    with col1:
        bmi = st.number_input("BMI", 10.0, 60.0, step=0.1)
        physical_health = st.number_input("Days Physically Unwell (past 30 days)", 0.0, 30.0, step=1.0)
        mental_health = st.number_input("Days Mentally Unwell (past 30 days)", 0.0, 30.0, step=1.0)
        sleep = st.number_input("Average Sleep Hours per Day", 0.0, 24.0, step=0.5)

    with col2:
        smoking = st.selectbox("Smoking Status", ["Yes", "No"])
        alcohol = st.selectbox("Drinks Alcohol", ["Yes", "No"])
        stroke = st.selectbox("History of Stroke", ["Yes", "No"])
        diff_walking = st.selectbox("Difficulty Walking", ["Yes", "No"])
        sex = st.selectbox("Sex", ["Male", "Female"])
        age = st.selectbox("Age Category", [
            '18-24', '25-29', '30-34', '35-39', '40-44', '45-49',
            '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80+'
        ])
        race = st.selectbox("Race", ['White', 'Black', 'Asian', 'American Indian/Alaskan Native', 'Other', 'Hispanic'])
        diabetic = st.selectbox("Diabetic Status", ['Yes', 'No', 'No, borderline diabetes', 'Yes (during pregnancy)'])
        activity = st.selectbox("Physically Active", ['Yes', 'No'])
        gen_health = st.selectbox("General Health", ['Poor', 'Fair', 'Good', 'Very good', 'Excellent'])
        asthma = st.selectbox("Has Asthma", ['Yes', 'No'])
        kidney = st.selectbox("Kidney Disease", ['Yes', 'No'])
        skin = st.selectbox("Skin Cancer", ['Yes', 'No'])

    submitted = st.form_submit_button("Run Prediction")

# ----------- Prediction Logic -----------
if submitted:
    patient_data = pd.DataFrame([{
        'BMI': bmi,
        'PhysicalHealth': physical_health,
        'MentalHealth': mental_health,
        'SleepTime': sleep,
        'Smoking': smoking,
        'AlcoholDrinking': alcohol,
        'Stroke': stroke,
        'DiffWalking': diff_walking,
        'Sex': sex,
        'AgeCategory': age,
        'Race': race,
        'Diabetic': diabetic,
        'PhysicalActivity': activity,
        'GenHealth': gen_health,
        'Asthma': asthma,
        'KidneyDisease': kidney,
        'SkinCancer': skin
    }])

    prob = model.predict_proba(patient_data)[0][1]
    is_risk = prob >= threshold
    label = "🛑 AT RISK" if is_risk else "✅ NOT AT RISK"

    with st.container():
        st.subheader("Prediction Result")
        if is_risk:
            st.error(f"**{label}** — Confidence: {prob:.2%}")
        else:
            st.success(f"**{label}** — Confidence: {prob:.2%}")

        st.caption(f"Risk Threshold Used: {threshold:.2f}")

# ----------- Feature Importance (Optional) -----------
if st.checkbox("📊 View Feature Importance"):
    try:
        importance = model.named_steps['xgbclassifier'].feature_importances_
        feature_names = model.named_steps['columntransformer'].get_feature_names_out()
        cleaned_names = [name.split("__")[1] if "__" in name else name for name in feature_names]

        df_imp = pd.DataFrame({'Feature': cleaned_names, 'Importance': importance})
        top = df_imp.sort_values(by='Importance', ascending=False).head(10)

        fig, ax = plt.subplots()
        top.plot(kind='barh', x='Feature', y='Importance', ax=ax, legend=False, color='skyblue')
        ax.invert_yaxis()
        ax.set_title("Top 10 Influential Features")
        st.pyplot(fig)
    except Exception as e:
        st.warning("Feature importance could not be loaded.")
        st.text(str(e))
