import streamlit as st
import numpy as np

st.title("Heart Disease Prediction")

# User input fields
age = st.number_input("Age", min_value=1, max_value=120, value=25, step=1)
sex = st.selectbox("Sex", ["Female", "Male"])
cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120)
chol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", ["No", "Yes"])
restecg = st.selectbox("Resting ECG Results", ["Normal", "ST-T wave abnormality", "Left Ventricular Hypertrophy"])
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=50, max_value=250, value=150)
exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
oldpeak = st.slider("ST Depression (Oldpeak)", min_value=0.0, max_value=6.0, value=1.0, step=0.1)
slope = st.selectbox("Slope of Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])
ca = st.slider("Number of Major Vessels Colored", min_value=0, max_value=3, value=0)
thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])

# Convert categorical inputs to numerical values
sex = 1 if sex == "Male" else 0
cp = ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"].index(cp) + 1
fbs = 1 if fbs == "Yes" else 0
restecg = ["Normal", "ST-T wave abnormality", "Left Ventricular Hypertrophy"].index(restecg)
exang = 1 if exang == "Yes" else 0
slope = ["Upsloping", "Flat", "Downsloping"].index(slope)
thal = ["Normal", "Fixed Defect", "Reversible Defect"].index(thal) + 1

# Create input array
input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

# Predict button
if st.button("Predict"):
    # Replace with actual model prediction when ready
    # prediction = model.predict(input_data)
    prediction = np.random.choice([0, 1])  # Simulated result (0: No Disease, 1: Disease)

    # Show result
    if prediction == 1:
        st.error("High Risk: Possible Heart Disease Detected! Consult a doctor.")
    else:
        st.success("Low Risk: No Heart Disease Detected.")
