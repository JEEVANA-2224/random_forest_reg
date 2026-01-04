import streamlit as st
import pandas as pd
import joblib

# ==============================
# Load Saved Model
# ==============================
model = joblib.load("random_forest_tips_model.pkl")

# ==============================
# App Title
# ==============================
st.set_page_config(page_title="Tip Prediction App", layout="centered")
st.title("💰 Tip Prediction using Random Forest")
st.write("Predict restaurant tips based on customer details")

# ==============================
# User Inputs
# ==============================
total_bill = st.number_input("Total Bill Amount ($)", min_value=1.0, step=0.5)

sex = st.selectbox("Sex", ["Male", "Female"])
smoker = st.selectbox("Smoker", ["Yes", "No"])
day = st.selectbox("Day", ["Thur", "Fri", "Sat", "Sun"])
time = st.selectbox("Time", ["Lunch", "Dinner"])
size = st.number_input("Group Size", min_value=1, step=1)

# ==============================
# Prediction Button
# ==============================
if st.button("Predict Tip"):
    input_data = pd.DataFrame({
        "total_bill": [total_bill],
        "sex": [sex],
        "smoker": [smoker],
        "day": [day],
        "time": [time],
        "size": [size]
    })

    prediction = model.predict(input_data)

    st.success(f"💵 Predicted Tip Amount: ${prediction[0]:.2f}")
