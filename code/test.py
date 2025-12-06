import pandas as pd
import numpy as np
import joblib
import os

# ===============================
# Load saved model and encoders
# ===============================
model_path = r"C:\Users\vishn\Documents\car price\models\used_car_price_model.pkl"
encoders_path = r"C:\Users\vishn\Documents\car price\models\label_encoders.pkl"

model = joblib.load(model_path)
label_encoders = joblib.load(encoders_path)

# ===============================
# Example test data
# ===============================
# You can create a dictionary for one or multiple cars
# Columns must match training features:
# ['Brand', 'model', 'Year', 'Age', 'kmDriven', 'Transmission', 'Owner', 'FuelType']

test_data = [
    {
        "Brand": "Honda",
        "model": "City",
        "Year": 2010,
        "Age": 13,
        "kmDriven": "85,000 km",
        "Transmission": "Manual",
        "Owner": "first",
        "FuelType": "Petrol"
    },
    {
        "Brand": "Toyota",
        "model": "Innova",
        "Year": 2015,
        "Age": 8,
        "kmDriven": "1,20,000 km",
        "Transmission": "Manual",
        "Owner": "second",
        "FuelType": "Diesel"
    }
]

# Convert to DataFrame
df_test = pd.DataFrame(test_data)

# ===============================
# Clean numeric columns
# ===============================
df_test['kmDriven'] = df_test['kmDriven'].str.replace(',', '', regex=False)
df_test['kmDriven'] = df_test['kmDriven'].str.replace(' km', '', regex=False).astype(float)

# ===============================
# Encode categorical variables
# ===============================
categorical_columns = ['Brand', 'model', 'Transmission', 'Owner', 'FuelType']
for col in categorical_columns:
    le = label_encoders[col]
    df_test[col] = le.transform(df_test[col])

# ===============================
# Predict prices
# ===============================
predicted_prices = model.predict(df_test)

# Show predictions
for i, price in enumerate(predicted_prices):
    print(f"Car {i+1} predicted price: ₹ {int(price):,}")
