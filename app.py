import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# ---- Load and prepare the dataset ----
dataset = pd.read_csv("weight-height.csv")  # Change to your CSV filename

# Drop unwanted columns
X = dataset.drop(['Weight', 'Gender'], axis=1)  # Feature(s)
y = dataset['Weight']  # Target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=123)

# Train the model
lr = LinearRegression()
lr.fit(X_train, y_train)

# ---- Streamlit UI ----
st.title("🤖  Weight Prediction App")
st.write("Predict weight based on height and other features.")

# Input features dynamically from the user
user_input = {}
for feature in X.columns:
    user_input[feature] = st.number_input(f"Enter {feature}:", value=float(X[feature].mean()))

# Make prediction when button is clicked
if st.button("Predict Weight"):
    input_df = pd.DataFrame([user_input])
    prediction = lr.predict(input_df)[0]
    st.success(f"🎯 Predicted Weight: {prediction:.2f} kg")
