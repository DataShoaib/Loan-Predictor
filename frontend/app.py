import streamlit as st
import requests


# Config

API_URL = "http://127.0.0.1:8000/predict"  # FastAPI endpoint

st.set_page_config(page_title="Loan Approval Predictor", layout="centered")

st.title("🏦 Loan Approval Prediction")
st.write("Fill in the details below to predict loan approval status:")


# User Inputs

no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=20, value=0)
education = st.selectbox("Education Level", ["Graduate", "not_Graduate"])
self_employed = st.selectbox("Self Employed?", ["Yes", "No"])
income_annum = st.number_input("Annual Income (₹)", min_value=0, value=500000)
loan_amount = st.number_input("Loan Amount (₹)", min_value=0, value=200000)
loan_term = st.number_input("Loan Term (Months)", min_value=0, value=20)
cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, value=700)
residential_assets_value = st.number_input("Residential Assets Value (₹)", min_value=0, value=500000)
commercial_assets_value = st.number_input("Commercial Assets Value (₹)", min_value=0, value=0)
luxury_assets_value = st.number_input("Luxury Assets Value (₹)", min_value=0, value=0)
bank_asset_value = st.number_input("Bank Assets Value (₹)", min_value=0, value=100000)

# -----------------------
# Submit Button
# -----------------------
if st.button("Predict Loan Approval"):

    loan_to_income = loan_amount / income_annum if income_annum != 0 else 0

    payload = {
        "no_of_dependents": int(no_of_dependents),
        "education": education,
        "self_employed": self_employed,
        "income_annum": float(income_annum),
        "loan_amount": float(loan_amount),
        "loan_term": int(loan_term),
        "cibil_score": int(cibil_score),
        "residential_assets_value": float(residential_assets_value),
        "commercial_assets_value": float(commercial_assets_value),
        "luxury_assets_value": float(luxury_assets_value),
        "bank_asset_value": float(bank_asset_value),
        "loan_to_income": float(loan_to_income)
    }

    response = requests.post(API_URL, json=payload)

    if response.status_code == 200:
        result = response.json()
        st.success(f"✅ Prediction: {result["loan_approval_status"]}")
        st.info(f"Probability: {result["probablity"]}")
    else:
        st.error(response.json())
