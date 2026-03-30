import streamlit as st
import pandas as pd
import numpy as np
import joblib

kmeans=joblib.load("kmeans.pkl")
scaler=joblib.load("scaler.pkl")


st.title("Customer Segmentation App")
st.write("Enter customer details to predict the segment")

age=st.number_input("Age",min_value=18,max_value=100,value=35)
income=st.number_input("Income",min_value=0,max_value=200000,value=50000)
total_spending=st.number_input("Total Spending",min_value=0,max_value=5000,value=1000)
number_of_web_purchase=st.number_input("Number of Web Purchases",min_value=0,max_value=100,value=10)
number_of_store_purchase=st.number_input("Number of Store Purchases",min_value=0,max_value=100,value=10)
number_of_web_visits=st.number_input("Number of Web visits",min_value=0,max_value=50,value=3)
Recency=st.number_input("Recency(Number of Days since last Purchase)",min_value=0,max_value=365,value=30)

input_data=pd.DataFrame({
    "Age": [age],
    "Income":[income],
    "Total_spends":[total_spending],
    "NumWebPurchases":[number_of_web_purchase],
    "NumStorePurchases":[number_of_store_purchase],
    
    "NumWebVisitsMonth":[number_of_web_visits],
    "Recency":[Recency]

})

input_scaled=scaler.transform(input_data)

if st.button("Predict Segment"):
    segment=kmeans.predict(input_scaled)[0]
    st.success(f"Predicted segment:{segment}")
