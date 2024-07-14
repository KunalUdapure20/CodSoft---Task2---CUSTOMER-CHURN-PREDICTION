import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import os

# Define the paths to the pickle files
df_path = os.path.join('C:\\', 'Users', 'ASUS', 'Downloads', 'df.pkl')
model_path = os.path.join('C:\\', 'Users', 'ASUS', 'Downloads', 'abc1.pkl')

# Check if files exist
if not os.path.exists(df_path):
    st.error(f"No such file: '{df_path}'")
    raise FileNotFoundError(f"No such file: '{df_path}'")
if not os.path.exists(model_path):
    st.error(f"No such file: '{model_path}'")
    raise FileNotFoundError(f"No such file: '{model_path}'")

# Loading models
df = pickle.load(open(df_path, 'rb'))
abc1 = pickle.load(open(model_path, 'rb'))

# Initialize LabelEncoders
le_Geography = LabelEncoder()
le_gender = LabelEncoder()

# Fit LabelEncoders on training data
le_Geography.fit(df['Geography'])
le_gender.fit(df['Gender'])

# StandardScaler for numeric features
ss = StandardScaler()

# Prediction Function
def prediction(CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary):
    # Handle empty fields
    if any(value == '' for value in [CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary]):
        st.error("Please provide valid input for all fields.")
        return None
    
    # Encode categorical features
    try:
        geography_encoded = le_Geography.transform([Geography])[0]
        gender_encoded = le_gender.transform([Gender])[0]
    except ValueError as e:
        st.error(f"Error encoding categorical data: {e}")
        return None
    
    # Prepare features for prediction
    features = np.array([[float(CreditScore), geography_encoded, gender_encoded, float(Age), float(Tenure), float(Balance), float(NumOfProducts), float(HasCrCard), float(IsActiveMember), float(EstimatedSalary)]])
    features = ss.fit_transform(features)
    
    # Make prediction
    try:
        prediction = abc1.predict(features).reshape(-1)
    except ValueError as e:
        st.error(f"Error during prediction: {e}")
        return None
    
    return prediction[0]

# Web App Interface
st.title('Bank Customer Churn Prediction')
CreditScore = st.number_input('Credit Score')
Geography = st.text_input('Geography')
Gender = st.text_input('Gender')
Age = st.number_input('Age')
Tenure = st.number_input('Tenure')
Balance = st.number_input('Balance')
NumOfProducts = st.number_input('Products Number')
HasCrCard = st.number_input('Credit Card')
IsActiveMember = st.number_input('Active Member')
EstimatedSalary = st.number_input('Estimated Salary')

if st.button('Predict'):
    pred = prediction(CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary)

    if pred is not None:
        if pred == 1:
            st.write("The customer has left.")
        else:
            st.write("The customer is still active.")