# import necessary libraries
import streamlit as st
import pandas as pd
import joblib

st.title("vehicle insurance prediction")

# read the dataset to fill list values
df = pd.read_csv('train.csv')

# create input fields - categorical columns
Gender = st.selectbox("Gender", pd.unique(df['Gender']))
Vehicle_Age = st.selectbox("Vehicle_Age", pd.unique(df['Vehicle_Age']))
Vehicle_Damage = st.selectbox("Vehicle_Damage", pd.unique(df['Vehicle_Damage']))

# create input fields - numerical columns
Age = st.number_input("Enter Age")
Driving_License = st.number_input("Enter Driving_License")
Region_Code = st.number_input("Enter Region_Code")
Previously_Insured = st.number_input("Enter Previously_Insured")
Annual_Premium = st.number_input("Enter Annual_Premium")
Policy_Sales_Channel = st.number_input("Enter Policy_Sales_Channel")
Vintage = st.number_input("Enter Vintage")

# convert the input values to dict - left ones are the column names within the data frame and right ones are the variables declared above in the input fields
inputs = {
    "Gender": Gender,
    "Age": Age,
    "Driving_License": Driving_License,
    "Region_Code": Region_Code,
    "Previously_Insured": Previously_Insured,
    "Vehicle_Age": Vehicle_Age,
    "Vehicle_Damage": Vehicle_Damage,
    "Annual_Premium": Annual_Premium,
    "Policy_Sales_Channel": Policy_Sales_Channel,
    "Vintage": Vintage
}

# on click
if st.button("Predict"):
    # load the pickle model 
    model = joblib.load('vehicle_insurance_recommendation_pipeline_model.pkl')

    X_input = pd.DataFrame(inputs,index=[0])
    # predict the target using the loaded model
    prediction = model.predict(X_input)
    # display the result
    st.write("The predicted value is: ")
    st.write(prediction)
