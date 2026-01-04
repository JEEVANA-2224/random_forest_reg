import streamlit as st
import pandas as pd
import pickle

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="🏠 House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 House Price Prediction App")
st.write("Predict house prices using **Random Forest Regression**")

# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    with open("random_forest_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# ---------------- User Input ----------------
st.subheader("📥 Enter House Details")

# Example inputs (adjust names to match dataset)
city = st.selectbox("City", ["Delhi", "Mumbai"])
area = st.number_input("Area (sq ft)", min_value=100, max_value=10000, step=50)
bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, step=1)
bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, step=1)
age = st.number_input("House Age (years)", min_value=0, max_value=100, step=1)

# ---------------- Create Input DataFrame ----------------
input_data = pd.DataFrame({
    "City": [city],
    "area": [area],
    "bedrooms": [bedrooms],
    "bathrooms": [bathrooms],
    "age": [age]
})

# ---------------- Prediction ----------------
if st.button("🔮 Predict Price"):
    prediction = model.predict(input_data)[0]
    st.success(f"💰 Estimated House Price: ₹ {prediction:,.2f}")
