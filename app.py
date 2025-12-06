import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ===============================
# Set up relative paths
# ===============================
BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, "models", "used_car_price_model.pkl")
encoders_path = os.path.join(BASE_DIR, "models", "label_encoders.pkl")

# ===============================
# Load model and encoders
# ===============================
model = joblib.load(model_path)
label_encoders = joblib.load(encoders_path)

# ===============================
# Streamlit UI
# ===============================
st.title("Used Car Price Prediction")
st.write("Enter car details below to predict the selling price:")

# Dropdowns for categorical features
brand = st.selectbox("Brand", label_encoders['Brand'].classes_)
model_name = st.selectbox("Model", label_encoders['model'].classes_)
transmission = st.selectbox("Transmission", label_encoders['Transmission'].classes_)
owner = st.selectbox("Owner Type", label_encoders['Owner'].classes_)
fuel_type = st.selectbox("Fuel Type", label_encoders['FuelType'].classes_)

# Sliders / inputs for numeric features
year = st.slider("Year of Manufacture", min_value=1980, max_value=2025, value=2015)
age = st.slider("Car Age (years)", min_value=0, max_value=50, value=5)
km_driven = st.text_input("Kilometers Driven (e.g., 85,000 km)", " ")

# ===============================
# Prediction
# ===============================
if st.button("Predict Price"):
    try:
        # Clean kmDriven input
        km_driven_clean = float(km_driven.replace(',', '').replace(' km', ''))
        
        # Create DataFrame
        input_df = pd.DataFrame({
            'Brand': [brand],
            'model': [model_name],
            'Year': [year],
            'Age': [age],
            'kmDriven': [km_driven_clean],
            'Transmission': [transmission],
            'Owner': [owner],
            'FuelType': [fuel_type]
        })
        
        # Encode categorical features
        for col in ['Brand', 'model', 'Transmission', 'Owner', 'FuelType']:
            le = label_encoders[col]
            input_df[col] = le.transform(input_df[col])
        
        # Predict price
        price = model.predict(input_df)[0]
        st.success(f"Predicted Selling Price: ₹ {int(price):,}")
    
    except Exception as e:
        st.error(f"Error in prediction: {e}")
