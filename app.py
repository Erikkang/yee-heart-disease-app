import streamlit as st
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt

# ---------- Load model & metadata ----------
@st.cache_resource
def load_model_and_meta():
    pipe = joblib.load('model_pipeline.pkl')
    meta = json.load(open('feature_metadata.json'))
    return pipe, meta

pipe, meta = load_model_and_meta()
num_feats = meta['num_feats']
cat_feats = meta['cat_feats']

# ---------- Page Config ----------
st.set_page_config(
    page_title='Heart Disease Risk Predictor',
    layout='centered'
)

st.title("Heart Disease Risk Predictor")

# ---------- Sidebar instructions ----------
with st.sidebar:
    st.header("How to Use")
    st.markdown("""
    1. Fill in the patient's clinical and lifestyle details.
    2. Adjust the risk threshold if needed.
    3. Click **Predict** to view the risk status and confidence score.
    """)
    threshold = st.slider('Risk Threshold', 0.0, 1.0, 0.50, 0.01)

# ---------- Input Form ----------
with st.form("input_form"):
    st.subheader("Patient Information")

    st.markdown("### Numeric Details")
    bmi = st.number_input('BMI (Body Mass Index)', 10.0, 60.0, step=0.1)
    physical_health = st.number_input('Physical Health (days unwell in past 30)', 0.0, 30.0, step=1.0)
    mental_health = st.number_input('Mental Health (days unwell in past 30)', 0.0, 30.0, step=1.0)
    sleep_time = st.number_input('Sleep Time (average hours per day)', 0.0, 24.0, step=0.5)

    st.markdown("### Medical & Lifestyle Details")
    smoking = st.selectbox('Smoking', ['Yes', 'No'])
    alcohol = st.selectbox('Alcohol Drinking', ['Yes', 'No'])
    stroke = st.selectbox('Stroke History', ['Yes', 'No'])
    diff_walking = st.selectbox('Difficulty Walking', ['Yes', 'No'])
    sex = st.selectbox('Sex', ['Male', 'Female'])
    age_category = st.selectbox('Age Category', [
        '18-24','25-29','30-34','35-39','40-44','45-49',
        '50-54','55-59','60-64','65-69','70-74','75-79','80+'
    ])
    race = st.selectbox('Race', ['White', 'Black', 'Asian', 'American Indian/Alaskan Native', 'Other', 'Hispanic'])
    diabetic = st.selectbox('Diabetic Status', ['Yes', 'No', 'No, borderline diabetes', 'Yes (during pregnancy)'])
    physical_activity = st.selectbox('Physical Activity', ['Yes', 'No'])
    gen_health = st.selectbox('General Health', ['Poor', 'Fair', 'Good', 'Very good', 'Excellent'])
    asthma = st.selectbox('Asthma', ['Yes', 'No'])
    kidney_disease = st.selectbox('Kidney Disease', ['Yes', 'No'])
    skin_cancer = st.selectbox('Skin Cancer', ['Yes', 'No'])

    submitted = st.form_submit_button("Predict")

# ---------- Prediction Output ----------
if submitted:
    sample = pd.DataFrame([{ 
        'BMI': bmi,
        'PhysicalHealth': physical_health,
        'MentalHealth': mental_health,
        'SleepTime': sleep_time,
        'Smoking': smoking,
        'AlcoholDrinking': alcohol,
        'Stroke': stroke,
        'DiffWalking': diff_walking,
        'Sex': sex,
        'AgeCategory': age_category,
        'Race': race,
        'Diabetic': diabetic,
        'PhysicalActivity': physical_activity,
        'GenHealth': gen_health,
        'Asthma': asthma,
        'KidneyDisease': kidney_disease,
        'SkinCancer': skin_cancer
    }])

    proba = pipe.predict_proba(sample)[0][1]
    label = 'AT RISK' if proba >= threshold else 'Not At Risk'

    st.subheader("Prediction Result")
    if label == 'AT RISK':
        st.error(f'{label}  (Confidence: {proba:.2%})')
    else:
        st.success(f'{label}  (Confidence: {proba:.2%})')

    st.caption(f"Threshold used: {threshold:.2f}")

# ---------- Optional: Feature importance chart ----------
if st.checkbox("\U0001F4CA Show top 10 most important features"):
    try:
        importances = pipe.named_steps['clf'].feature_importances_
        raw_feat_names = pipe.named_steps['pre'].get_feature_names_out()

        clean_feat_names = [name.split("__")[1] if "__" in name else name for name in raw_feat_names]

        imp_df = pd.DataFrame({
            'feature': clean_feat_names,
            'importance': importances
        })

        top10 = imp_df.sort_values(by='importance', ascending=False).head(10)

        friendly_names = {
            "DiffWalking_Yes": "Difficulty Walking (Yes)",
            "Diabetic_Yes": "Diabetic",
            "GenHealth_Excellent": "Excellent General Health",
            "AgeCategory_65-69": "Age 65–69",
            "Stroke_Yes": "Had Stroke",
            "Smoking_Yes": "Smoker",
            "BMI": "Body Mass Index",
            "PhysicalHealth": "Physical Health (days)",
            "MentalHealth": "Mental Health (days)",
            "SleepTime": "Sleep Time (hrs)"
        }

        top10['feature'] = top10['feature'].map(lambda x: friendly_names.get(x, x))

        fig, ax = plt.subplots()
        top10.plot(kind='barh', x='feature', y='importance', ax=ax, legend=False, color='orange')
        ax.invert_yaxis()
        ax.set_title("Top 10 Most Important Features")
        st.pyplot(fig)

    except Exception as e:
        st.warning("Feature importance could not be displayed.")
        st.text(str(e))