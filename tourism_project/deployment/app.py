import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="rahulg1987/Tourism-Package-Prediction", filename="best_tourism_package_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Machine Failure Prediction
st.title("Tourism Package Prediction App")
st.write("""
This application predicts the likelihood of a Tourism Package Purchased on not based on its parameters.
Please enter the data below to get a prediction.
""")

# User input
TypeofContact = st.selectbox("TypeofContact", ["Self Enquiry", "Company Invited"])
Occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
Gender = st.selectbox("Gender", ["Male", "Female"])
ProductPitched = st.selectbox("ProductPitched", ["Basic", "Deluxe", "Standard", "King", "Super Deluxe"])
MaritalStatus = st.selectbox("MaritalStatus", ["Married", "Unmarried", "Single", "Divorced"])
Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "VP", "AVP"])
Age = st.number_input("Age", min_value=18, max_value=90, value=21, step=1)
CityTier = st.number_input("CityTier", min_value=1, max_value=3, value=1, step=1)
DurationOfPitch = st.number_input("DurationOfPitch", min_value=6.0, max_value=100.0, value=6.0, step=1)
NumberOfPersonVisiting = st.number_input("NumberOfPersonVisiting", min_value=2, max_value=5, value=2, step=1)
NumberOfFollowups = st.number_input("NumberOfFollowups", min_value=2.0, max_value=5.0, value=2.0, step=1)
PreferredPropertyStar = st.number_input("PreferredPropertyStar", min_value=1.0, max_value=5.0, value=1.0, step=1)
NumberOfTrips = st.number_input("NumberOfTrips", min_value=1.0, max_value=10.0, value=1.0, step=1)
PitchSatisfactionScore = st.number_input("PitchSatisfactionScore", min_value=1, max_value=5, value=1, step=1)
NumberOfChildrenVisiting = st.number_input("NumberOfChildrenVisiting", min_value=0.0, max_value=2.0, value=0.0, step=1)
MonthlyIncome = st.number_input("NumberOfTrips", min_value=1000.0, max_value=50000.0, value=20000.0, step=1)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'TypeofContact': TypeofContact,
    'Occupation': Occupation,
    'Gender': Gender,
    'ProductPitched': ProductPitched,
    'MaritalStatus': MaritalStatus,
    'Designation': Designation,
    'Age': Age,
    'CityTier': CityTier,
    'DurationOfPitch': DurationOfPitch,
    'ProductPitched': ProductPitched,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'PreferredPropertyStar': PreferredPropertyStar,
    'NumberOfTrips': NumberOfTrips,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'MonthlyIncome': MonthlyIncome
}])


if st.button("Predict Purchase"):
    prediction = model.predict(input_data)[0]
    result = "Package Purchase" if prediction == 1 else "Package Not Purchase"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
