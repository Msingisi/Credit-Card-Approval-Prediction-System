import os
import streamlit as st
import requests
import pandas as pd
import numpy as np
from streamlit_lottie import st_lottie_spinner
from dotenv import load_dotenv

load_dotenv()

# --- Page Config ---
st.set_page_config(page_title="Credit Risk Predictor", layout="centered")
st.title("Credit Card Approval Prediction")
st.markdown("Enter the applicant information below:")

# --- Load Lottie animation ---
@st.cache_data
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Lottie animation URL
lottie_spinner = load_lottieurl("https://assets4.lottiefiles.com/packages/lf20_usmfx6bp.json")

# --- Form Input Fields ---
with st.form("input_form"):
    age = np.negative(st.slider("Age (years)", value=42, min_value=18, max_value=70, step=1) * 365.25)
    income = int(st.text_input("Income (USD)", 0))
    emp_length = np.negative(st.slider("Employment Length (years)", value=6, min_value=0, max_value=30, step=1)* 365.25)
    family_members = float(st.selectbox("Family Member Count", [1, 2, 3, 4, 5, 6]))

    education = st.selectbox("Education Level", [
        "Lower secondary", "Secondary / secondary special", "Incomplete higher",
        "Higher education", "Academic degree"
    ])
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", [
        "Married", "Single / not married", "Separated", "Widow", "Civil marriage"
    ])
    dwelling = st.selectbox("Dwelling Type", [
        "House / apartment", "With parents", "Municipal apartment",
        "Rented apartment", "Office apartment", "Co-op apartment"
    ])
    employment_status = st.selectbox("Employment Status", [
        "Working", "Commercial associate", "Pensioner", "State servant", "Student"
    ])
    has_car = st.selectbox("Owns a Car?", ["Yes", "No"])
    has_property = st.selectbox("Owns Property?", ["Yes", "No"])
    has_phone = st.selectbox("Has Phone?", ["Yes", "No"])
    has_email = st.selectbox("Has Email?", ["Yes", "No"])
    has_work_phone = st.selectbox("Has Work Phone?", ["Yes", "No"])

    submitted = st.form_submit_button("Predict")

# --- On Submit ---
if submitted:
    data = {
        'Age': age,
        'Income': income,
        'Employment length': emp_length,
        'Family member count': family_members,
        'Education level': education,
        'Gender': gender,
        'Marital status': marital_status,
        'Dwelling': dwelling,
        'Employment status': employment_status,
        'Has a car': has_car,
        'Has a property': has_property,
        'Has a phone': has_phone,
        'Has an email': has_email,
        'Has a work phone': has_work_phone,
    }

    df_input = pd.DataFrame([data])

    # Ordinal Encoding for Education level
    edu_order = ['Lower secondary', 'Secondary / secondary special', 'Incomplete higher', 'Higher education', 'Academic degree']
    edu_map = {label: idx for idx, label in enumerate(edu_order)}
    df_input['Education level'] = df_input['Education level'].map(edu_map)

    # One-hot encoding
    cat_cols = ['Gender', 'Marital status', 'Dwelling', 'Employment status',
                'Has a car', 'Has a property', 'Has a phone', 'Has an email', 'Has a work phone']
    expected_columns = [
        'Gender_Female', 'Gender_Male',
        'Marital status_Civil marriage', 'Marital status_Married', 'Marital status_Separated',
        'Marital status_Single / not married', 'Marital status_Widow',
        'Dwelling_Co-op apartment', 'Dwelling_House / apartment', 'Dwelling_Municipal apartment',
        'Dwelling_Office apartment', 'Dwelling_Rented apartment', 'Dwelling_With parents',
        'Employment status_Commercial associate', 'Employment status_Pensioner',
        'Employment status_State servant', 'Employment status_Student', 'Employment status_Working',
        'Has a car_No', 'Has a car_Yes',
        'Has a property_No', 'Has a property_Yes',
        'Has a phone_No', 'Has a phone_Yes',
        'Has an email_No', 'Has an email_Yes',
        'Has a work phone_No', 'Has a work phone_Yes'
    ]

    df_input = pd.get_dummies(df_input, columns=cat_cols)
    for col in expected_columns:
        if col not in df_input.columns:
            df_input[col] = 0

    df_input = df_input[['Income', 'Education level', 'Age', 'Employment length', 'Family member count'] + expected_columns]

    # --- Load prediction endpoint from environment variable ---
    
    model_url = os.getenv("MLFLOW_MODEL_URL", "http://127.0.0.1:5005/invocations")

    # --- Predict ---
    try:
        with st_lottie_spinner(lottie_spinner, height=180):
            response = requests.post(url=model_url,
                headers={"Content-Type": "application/json"},
                json={"dataframe_records": df_input.to_dict(orient="records")}
            )

        if response.status_code == 200:
            pred = response.json()['predictions'][0]
            st.success(f"Prediction: {'Credit card approved (Low Risk)' if pred == 0 else 'Credit card denied (High Risk)'}")
        else:
            st.error(f"Prediction failed: {response.status_code}\n{response.text}")
    except Exception as e:
        st.error(f"Connection error: {str(e)}")