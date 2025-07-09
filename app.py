import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load('C:/Users/Rishabh Singh/customer-churn-prediction/model/best_churn_model.pkl')

# Expected columns from training
expected_columns = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'PaperlessBilling', 'MonthlyCharges', 'TotalCharges',
    'Contract_One year', 'Contract_Two year',
    'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check',
    'PaymentMethod_Mailed check'
]

st.title("📊 Telco Customer Churn Prediction App")

# Input fields
gender = st.selectbox('Gender', ['Male', 'Female'])
SeniorCitizen = st.selectbox('Senior Citizen', [0, 1])
Partner = st.selectbox('Partner', ['Yes', 'No'])
Dependents = st.selectbox('Dependents', ['Yes', 'No'])
tenure = st.slider('Tenure (months)', 0, 72, 12)
PhoneService = st.selectbox('Phone Service', ['Yes', 'No'])
MultipleLines = st.selectbox('Multiple Lines', ['Yes', 'No', 'No phone service'])
InternetService = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
OnlineSecurity = st.selectbox('Online Security', ['Yes', 'No', 'No internet service'])
OnlineBackup = st.selectbox('Online Backup', ['Yes', 'No', 'No internet service'])
DeviceProtection = st.selectbox('Device Protection', ['Yes', 'No', 'No internet service'])
TechSupport = st.selectbox('Tech Support', ['Yes', 'No', 'No internet service'])
StreamingTV = st.selectbox('Streaming TV', ['Yes', 'No', 'No internet service'])
StreamingMovies = st.selectbox('Streaming Movies', ['Yes', 'No', 'No internet service'])
PaperlessBilling = st.selectbox('Paperless Billing', ['Yes', 'No'])
MonthlyCharges = st.number_input('Monthly Charges', 0.0, 200.0, 70.0)
TotalCharges = st.number_input('Total Charges', 0.0, 10000.0, 2500.0)
Contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
PaymentMethod = st.selectbox('Payment Method', [
    'Electronic check',
    'Mailed check',
    'Bank transfer (automatic)',
    'Credit card (automatic)'
])

if st.button('Predict'):
    # Create initial dataframe
    input_dict = {
        'gender': [gender],
        'SeniorCitizen': [SeniorCitizen],
        'Partner': [Partner],
        'Dependents': [Dependents],
        'tenure': [tenure],
        'PhoneService': [PhoneService],
        'MultipleLines': [MultipleLines],
        'InternetService': [InternetService],
        'OnlineSecurity': [OnlineSecurity],
        'OnlineBackup': [OnlineBackup],
        'DeviceProtection': [DeviceProtection],
        'TechSupport': [TechSupport],
        'StreamingTV': [StreamingTV],
        'StreamingMovies': [StreamingMovies],
        'PaperlessBilling': [PaperlessBilling],
        'MonthlyCharges': [MonthlyCharges],
        'TotalCharges': [TotalCharges],
        'Contract': [Contract],
        'PaymentMethod': [PaymentMethod]
    }

    input_df = pd.DataFrame(input_dict)

    # One-hot encode
    input_encoded = pd.get_dummies(input_df)

    # Add missing columns
    for col in expected_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    # Reorder to match model input
    input_encoded = input_encoded[expected_columns]

    # Predict
    prediction = model.predict(input_encoded)[0]

    # Display
    st.subheader("Prediction:")
    if prediction == 1 or prediction == 'Yes':
        st.error("⚠ The customer is likely to churn.")
    else:
        st.success("✅ The customer is not likely to churn.")
