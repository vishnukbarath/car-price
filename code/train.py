import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import re

# ===============================
# Load dataset
# ===============================
data_path = r"C:\Users\vishn\Documents\cloud\sd\data\used_cars_dataset_v2.csv"
df = pd.read_csv(data_path)

# ===============================
# Data cleaning
# ===============================

# Clean kmDriven: remove commas, " km", convert to float
df['kmDriven'] = df['kmDriven'].astype(str).str.replace(',', '', regex=False)
df['kmDriven'] = df['kmDriven'].str.replace(' km', '', regex=False).astype(float)

# Clean AskPrice: remove ₹ and commas, convert to float
df['AskPrice'] = df['AskPrice'].astype(str).str.replace('₹', '', regex=False)
df['AskPrice'] = df['AskPrice'].str.replace(',', '', regex=False).astype(float)

# Convert numeric columns with units if they exist
if 'mileage' in df.columns:
    df['mileage'] = df['mileage'].str.split(' ').str[0].str.replace(',', '').astype(float)
if 'engine' in df.columns:
    df['engine'] = df['engine'].str.split(' ').str[0].str.replace(',', '').astype(float)
if 'max_power' in df.columns:
    df['max_power'] = df['max_power'].str.split(' ').str[0].str.replace(',', '').astype(float)

# Drop irrelevant columns
df = df.drop(columns=['PostedDate', 'AdditionInfo'], errors='ignore')

# Drop rows with missing values
df.dropna(inplace=True)

# ===============================
# Encode categorical variables
# ===============================
categorical_columns = ['Brand', 'model', 'Transmission', 'Owner', 'FuelType']
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# ===============================
# Features and target
# ===============================
X = df.drop(columns=['AskPrice'])
y = df['AskPrice']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===============================
# Train Random Forest Regressor
# ===============================
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# ===============================
# Evaluate model
# ===============================
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R2 Score: {r2}")

# ===============================
# Save model and encoders
# ===============================
model_dir = r"C:\Users\vishn\Documents\car price\models"
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "used_car_price_model.pkl")
joblib.dump(model, model_path)

encoders_path = os.path.join(model_dir, "label_encoders.pkl")
joblib.dump(label_encoders, encoders_path)

print(f"Model saved at: {model_path}")
print(f"Label encoders saved at: {encoders_path}")
