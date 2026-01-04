import streamlit as st
import pandas as pd
import pickle

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="🏠 House Price Prediction",
    page_icon="🏡",
    layout="centered"
)

st.title("🏠 House Price Prediction App")
st.write("Predict house prices using a trained Random Forest model.")

# ---------------- Load Trained Pipeline ----------------
@st.cache_resource
def load_pipeline():
    with open("random_forest_model.pkl", "rb") as f:
        return pickle.load(f)

pipeline = load_pipeline()

# ---------------- User Inputs ----------------
st.header("Enter House Details")

def user_input_features():
    data = {
        'house_type': st.selectbox("House Type", ["Detached", "Semi-Detached", "Apartment"]),
        'location': st.text_input("Location", "Downtown"),
        'year_built': st.number_input("Year Built", 1900, 2026, 2000),
        'lot_size': st.number_input("Lot Size (sq ft)", 100, 100000, 5000),
        'sqft': st.number_input("Total Sqft", 100, 10000, 2500),
        'has_pool': st.selectbox("Has Pool", [0,1]),
        'condition': st.selectbox("Condition", ["Poor", "Average", "Good", "Excellent"]),
        'garage': st.number_input("Garage Spaces", 0, 10, 2),
        'has_basement': st.selectbox("Has Basement", [0,1]),
        'school_rating': st.number_input("Nearby School Rating (1-10)", 1, 10, 8),
        'has_fireplace': st.selectbox("Has Fireplace", [0,1])
    }
    return pd.DataFrame([data])

input_df = user_input_features()

# ---------------- Align Columns ----------------
expected_cols = pipeline.named_steps['preprocess'].feature_names_in_
for col in expected_cols:
    if col not in input_df.columns:
        # Fill numeric missing columns with 0, categorical with empty string
        if col in input_df.select_dtypes(include=["int64","float64"]).columns:
            input_df[col] = 0
        else:
            input_df[col] = ''

# ---------------- Prediction ----------------
if st.button("Predict Price"):
    prediction = pipeline.predict(input_df)[0]
    st.success(f"🏷 Predicted House Price: ₹{prediction:,.2f}")
