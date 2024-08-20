import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit App
st.title("Heart Disease Prediction")

st.write("""
This app predicts whether a person has heart disease based on input features.
""")

# Collect user input features
age = st.number_input('Age', min_value=0, max_value=120, value=30)
sex = st.selectbox('Sex', ('Male', 'Female'))
cp = st.selectbox('Chest Pain Type (cp)', (0, 1, 2, 3))
trestbps = st.number_input('Resting Blood Pressure (trestbps)', min_value=80, max_value=200, value=120)
chol = st.number_input('Serum Cholestoral in mg/dl (chol)', min_value=100, max_value=600, value=200)
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs)', (0, 1))
restecg = st.selectbox('Resting Electrocardiographic Results (restecg)', (0, 1, 2))
thalach = st.number_input('Maximum Heart Rate Achieved (thalach)', min_value=60, max_value=220, value=150)
exang = st.selectbox('Exercise Induced Angina (exang)', (0, 1))
oldpeak = st.number_input('ST Depression Induced by Exercise (oldpeak)', min_value=0.0, max_value=10.0, step=0.1, value=1.0)
slope = st.selectbox('Slope of the Peak Exercise ST Segment (slope)', (0, 1, 2))
ca = st.selectbox('Number of Major Vessels Colored by Flourosopy (ca)', (0, 1, 2, 3, 4))
thal = st.selectbox('Thalassemia (thal)', (0, 1, 2, 3))

# Convert categorical data to numerical
sex = 1 if sex == 'Male' else 0

# Make prediction
if st.button('Predict'):
    # Prepare the feature vector for prediction
    features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    prediction = model.predict(features)

    # Display the result
    if prediction[0] == 1:
        st.error("The model predicts that this person **HAS** heart disease.")
    else:
        st.success("The model predicts that this person **DOES NOT HAVE** heart disease.")

# Additional information
st.write("This model uses features like age, blood pressure, cholesterol levels, and others to predict the likelihood of heart disease.")