import streamlit as st
import pickle
import numpy as np

# Load the trained model, scaler, and the newly saved label encoders
model = pickle.load(open('heart_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
encoders = pickle.load(open('label_encoders.pkl', 'rb'))

st.title("🩺 Cardiovascular Risk Prediction System")
st.write("Enter the patient's clinical and demographic features to predict cardiovascular risk.")

# Create input layout
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sex", ['M', 'F'])
    cp = st.selectbox("Chest Pain Type", ['ATA', 'NAP', 'ASY', 'TA'])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120)
    chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=0, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])

with col2:
    restecg = st.selectbox("Resting ECG Results", ['Normal', 'ST', 'LVH'])
    thalach = st.number_input("Maximum Heart Rate Achieved", min_value=50, max_value=250, value=150)
    exang = st.selectbox("Exercise Induced Angina", ['N', 'Y'])
    oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=-3.0, max_value=10.0, value=1.0)
    slope = st.selectbox("ST Slope", ['Up', 'Flat', 'Down'])

if st.button("Predict Risk"):
    try:
        # Transform the user's text inputs into numbers using the saved encoders
        sex_enc = encoders['Sex'].transform([sex])[0]
        cp_enc = encoders['ChestPainType'].transform([cp])[0]
        restecg_enc = encoders['RestingECG'].transform([restecg])[0]
        exang_enc = encoders['ExerciseAngina'].transform([exang])[0]
        slope_enc = encoders['ST_Slope'].transform([slope])[0]

        # Combine all 11 inputs into the exact order the model expects
        user_data = np.array([[age, sex_enc, cp_enc, trestbps, chol, fbs, restecg_enc, thalach, exang_enc, oldpeak, slope_enc]])

        # Scale the data
        scaled_data = scaler.transform(user_data)

        # Make prediction
        prediction = model.predict(scaled_data)
        probability = model.predict_proba(scaled_data)[0][1] if hasattr(model, "predict_proba") else None

        st.markdown("---")
        if prediction[0] == 1:
            if probability is not None:
                st.error(f"⚠️ **High Risk Detected.** (Confidence: {probability*100:.2f}%)")
            else:
                 st.error("⚠️ **High Risk Detected.** The model predicts a high likelihood of cardiovascular disease.")
        else:
            if probability is not None:
                st.success(f"✅ **Low Risk.** (Confidence: {(1-probability)*100:.2f}%)")
            else:
                 st.success("✅ **Low Risk.** The model predicts a low likelihood of cardiovascular disease.")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
