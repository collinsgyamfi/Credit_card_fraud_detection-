import streamlit as st
import pandas as pd
import joblib
import lightgbm as lgb 
from geopy.distance import geodesic

model = joblib.load("fraud_detection_model.jb")
encoder = joblib.load("label_encoders.jb")

def haversine(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1),(lat2,lon2)).km

st.image("Fraud detection.jpeg", width=600)
st.title("Credit Card Fraud Detection System")
st.write("Credit card fraud poses significant challenges for financial institutions and consumers, especially as digital transactions increase. To combat evolving fraud tactics, machine learning (ML) has become essential, offering advanced methods for real-time detection and prevention of fraudulent activities, thereby enhancing security and maintaining consumer trust in digital payments.")

st.write("Please enter the transaction details below to check for fraud.")


merchant = st.text_input("Merchant Name")
category = st.text_input("Category")
amt = st.number_input("Transaction Amount", min_value=0.0, format="%.2f")
lat = st.number_input("Latitude",format="%.6f")
long = st.number_input("Longitude",format="%.6f")
merch_lat = st.number_input("Merchant Latitude",format="%.6f")
merch_long = st.number_input("Merchant Longitude",format="%.6f")
hour = st.slider("Transaction Hour",0,23,12)
day =st.slider("Transaction Day",1,31,15)
month = st.slider("Transaction MOnth",1,12,6)
gender = st.selectbox("Gender",["Male","Female"])
cc_num = st.text_input("Credit Card number")

distance = haversine(lat,long,merch_lat,merch_long)

if st.button("Check For Fraud"):
    if merchant and category and cc_num:
        input_data = pd.DataFrame([[merchant, category,amt,distance,hour,day,month,gender, cc_num]],
                                  columns=['merchant','category','amt','distance','hour','day','month','gender','cc_num'])
        
        categorical_col = ['merchant','category','gender']
        for col in categorical_col:
            try:
                input_data[col] = encoder[col].transform(input_data[col])
            except ValueError:
                input_data[col]=-1

        input_data['cc_num'] = input_data['cc_num'].apply(lambda x:hash(x) % (10 ** 2))
        prediction = model.predict(input_data)[0]
        result = "Fraudulant Transaction" if prediction == 1 else " Legitimate Transaction"
        st.subheader(f"Prediction: {result}")
    else:
        st.error("Please Fill all required fields")




st.write("This app was developed to help detect fraudulent transactions using machine learning techniques. It is important to note that while the model can provide valuable insights, it should not be solely relied upon for making financial decisions. Always exercise caution and verify transactions through trusted channels.")  

st.write("Developed by Dr.Ing Collins (Health data scientist)")