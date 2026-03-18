import streamlit as st
import joblib
import pandas as pd

# load model
kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Customer Segmentation App")

# inputs

income = st.number_input("Income", step=1)
age = st.number_input("Age", step=1)
spend = st.number_input("Total Spend", step=1)
recency = st.number_input("Recency", step=1)
web_purchases = st.number_input("Web Purchases", step=1)
store_purchases = st.number_input("Store Purchases", step=1)
web_visits = st.number_input("Web Visits per Month", step=1)
# button
if st.button("Predict"):
    data = pd.DataFrame({
        "Income":[income],
        "age":[age],
        "total_spend":[spend],
        "Recency":[recency],
        "NumWebPurchases":[web_purchases],
        "NumStorePurchases":[store_purchases],
        "NumWebVisitsMonth":[web_visits]
    })

    scaled = scaler.transform(data)
    cluster = kmeans.predict(scaled)

    st.success(f"Cluster: {cluster[0]}")