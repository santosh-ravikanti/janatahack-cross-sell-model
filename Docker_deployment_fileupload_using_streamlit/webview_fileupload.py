# import necessary libraries
import streamlit as st
import pandas as pd
import joblib

st.title("Promotion prediction")

# read the dataset to fill list values
df = pd.read_csv('train_LZdllcl.csv')

# create input fields - categorical columns
department = st.selectbox("department", pd.unique(df['department']))
region = st.selectbox("region", pd.unique(df['region']))
education = st.selectbox("education", pd.unique(df['education']))
gender = st.selectbox("gender", pd.unique(df['gender']))
recruitment_channel = st.selectbox("recruitment_channel", pd.unique(df['recruitment_channel']))

# create input fields - numerical columns
no_of_trainings = st.number_input("no_of_trainings")
previous_year_rating = st.number_input("previous_year_rating")
length_of_service = st.number_input("length_of_service")
KPIs_met_80 = st.number_input("Enter KPIs_met")
awards_won = st.number_input("awards_won")
average_training_score = st.number_input("avg_training_score")
age = st.number_input("age")

# convert the input values to dict - left ones are the column names within the data frame and right ones are the variables declared above in the input fields
inputs = {
  "department": department,
  "region": region,
  "gender": gender,
  "education": education,
  "recruitment_channel": recruitment_channel,
  "no_of_trainings": no_of_trainings,
  "age": age,  
  "previous_year_rating": previous_year_rating,
  "length_of_service": length_of_service,
  "KPIs_met >80%": KPIs_met_80,
  "awards_won?": awards_won,
  "avg_training_score": average_training_score
}

# on click
if st.button("Predict"):
    # load the pickle model 
    model = joblib.load('promote_pipeline_model.pkl')

    X_input = pd.DataFrame(inputs,index=[0])
    # predict the target using the loaded model
    prediction = model.predict(X_input)
    # display the result
    st.write("The predicted value is: ")
    st.write(prediction)

#File upload - to pass multiple values; like test file prediction
st.subheader("Please upload a csv file for prediction")
upload_file = st.file_uploader("Choose a csv file: ", type=['csv'])

if upload_file is not None:
    df = pd.read_csv(upload_file)

    st.write("file uploaded successfully; sample two records below")
    st.write(df.head(2))
    model = joblib.load('promote_pipeline_model.pkl')
    
    if st.button("Predict for the uploaded file"):
        df['is_promoted'] = model.predict(df)
        st.write("Prediction completed")
        st.write(df.head(2))        
        st.download_button(label = 'Download Predicted result',
                          data = df.to_csv(index=False),
                          file_name="Predictions.csv",
                          mime="text/csv")