import streamlit as st
import requests
import logging
import os

# ----------------- LOGGING CONFIGURATION -----------------
os.makedirs("logs", exist_ok=True)  # Create logs folder if missing
logging.basicConfig(
    filename="logs/streamlit.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

API_URL = "http://localhost:8000"

st.title("Bank Customer Churn Prediction")

# ----------------- LOGIN FORM -----------------
st.subheader("Login")
userid = st.text_input("User ID")
password = st.text_input("Password", type="password")

if st.button("Login"):
    try:
        response = requests.post(f"{API_URL}/login", json={"userid": userid, "password": password})
        if response.status_code == 200:
            st.write(response.text)
            logging.info(f"Login successful for UserID: {userid}")
        else: 
            st.error("Login request failed.")
            logging.warning(f"Login failed for UserID: {userid}")
    except Exception as e:
        st.error("Error connecting to backend.")
        logging.error(f"Error during login request: {str(e)}")

# ----------------- CHURN PREDICTION FORM -----------------
st.subheader("Enter Customer Details")

CreditScore = st.number_input("Credit Score", min_value=0)
Age = st.number_input("Age", min_value=0)
Tenure = st.number_input("Tenure", min_value=0)
Balance = st.number_input("Balance", min_value=0)
NumOfProducts = st.number_input("Number of Products", min_value=0)
HasCrCard = st.selectbox("Has Credit Card?", [0, 1])
IsActiveMember = st.selectbox("Is Active Member?", [0, 1])
stimatedSalary = st.number_input("Estimated Salary", min_value=0)
Geography_encoded = st.selectbox("Geography Encoded", [0, 1, 2])
Gender_encoded = st.selectbox("Gender Encoded", [0, 1])

if st.button("Predict Churn"):
    try:
        input_data = {
            "CreditScore": CreditScore,
            "Age": Age,
            "Tenure": Tenure,
            "Balance": Balance,
            "NumOfProducts": NumOfProducts,
            "HasCrCard": HasCrCard,
            "IsActiveMember": IsActiveMember,
            "stimatedSalary": stimatedSalary,
            "Geography_encoded": Geography_encoded,
            "Gender_encoded": Gender_encoded
        }
        response = requests.post(f"{API_URL}/input", json=input_data)
        
        if response.status_code == 200:
            result = response.json()
            st.success(f"Prediction Result: {result}")
            logging.info(f"Prediction successful: {result}")
        else:
            st.error("Prediction request failed.")
            logging.warning(f"Prediction failed. Status code: {response.status_code}")
    except Exception as e:
        st.error("Error connecting to backend.")
        logging.error(f"Error during prediction request: {str(e)}")
