#This is the code for the GUI implementation of pipeline.py
import streamlit as st
import pandas as pd
import joblib
from pipeline import processData, joinCensus, joinFema, createFeatures, generatePreds  # Import your processing functions

# Setting up the page
st.title('Business Closure Prediction Tool')

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

# User inputs
day_of_week = st.text_input("Enter the day of the week:")
day_of_disaster = st.number_input("Enter the day of disaster:", step=1, format='%d')
usr_thresh = usr_thresh = st.number_input("Enter the prediction threshold:", step=0.01, value=0.5)

if st.button('Predict'):
    if uploaded_file is not None and day_of_week and day_of_disaster:
        # Process the data through your pipeline
        data['Day of Week'] = day_of_week
        data['Day of Disaster'] = day_of_disaster
        processed_data = processData(data, day_of_week, day_of_disaster)
        census_data = joinCensus(processed_data)
        fema_data = joinFema(census_data)
        feature_data = createFeatures(fema_data)
        predictions = generatePreds(feature_data, usr_thresh)
        
        # Convert DataFrame to CSV
        result = predictions.to_csv(index=False)
        st.download_button(label="Download data as CSV",
                           data=result,
                           file_name='predicted_closures.csv',
                           mime='text/csv')
    else:
        st.error("Please upload a CSV file and fill out all input boxes.")

