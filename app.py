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

# cluster interpretation (based on your analysis)
cluster_info = {
    0: {
        "title": "High Income – High Spending 💎",
        "desc": "These customers have high income and spend a lot. They are your most valuable segment and should be targeted with premium offers and loyalty programs."
    },
    1: {
        "title": "Low Income – Low Spending 📉",
        "desc": "These customers have limited income and low spending behavior. Focus on affordable offers and discounts to engage them."
    },
    2: {
        "title": "Young Customers 🎯",
        "desc": "This segment represents younger customers. They may be more responsive to trends, social media campaigns, and modern marketing strategies."
    },
    3: {
        "title": "Frequent Buyers 🔁",
        "desc": "These customers purchase frequently. They are loyal and engaged, so consider retention strategies and exclusive deals."
    }
}

# button
if st.button("Predict"):
    data = pd.DataFrame({
        "Income": [income],
        "age": [age],
        "total_spend": [spend],
        "Recency": [recency],
        "NumWebPurchases": [web_purchases],
        "NumStorePurchases": [store_purchases],
        "NumWebVisitsMonth": [web_visits]
    })

    scaled = scaler.transform(data)
    cluster = kmeans.predict(scaled)[0]

    # show result
    st.success(f"Predicted Cluster: {cluster}")

    # show interpretation
    if cluster in cluster_info:
        st.subheader(cluster_info[cluster]["title"])
        st.write(cluster_info[cluster]["desc"])