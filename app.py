import numpy as np
import streamlit as st
import joblib
from sklearn.metrics import accuracy_score,  confusion_matrix

# Load the model and pre-processing objects
logistic_model = joblib.load('logisticRegression.pkl')
standard_scaler = joblib.load('StandardScaler.pkl')
label_encoder_severity = joblib.load('encoder_incident_severity.pkl')
one_hot_encoder = joblib.load('OneHotEncoder.pkl')

st.title('Insurance Claims - Fraud Detection')
st.markdown("### Please enter the details below to check if a claim is fraudulent.")

def user_input_features():
    st.sidebar.header('User Input Features')
    
    # Collecting inputs from user
    incident_hour_of_the_day = float(st.sidebar.slider('Incident Hour of the Day', 0, 23, 5))
    number_of_vehicles_involved = float(st.sidebar.slider('Number of Vehicles Involved', 1, 4, 1))
    bodily_injuries = float(st.sidebar.slider('Bodily Injuries', 0, 4, 1))
    witnesses = float(st.sidebar.slider('Witnesses', 0, 4, 1))
    incident_year = float(st.sidebar.selectbox('Incident Year', list(range(2015, 2021))))
    incident_month = float(st.sidebar.selectbox('Incident Month', list(range(1, 13))))
    incident_day = float(st.sidebar.selectbox('Incident Day', list(range(1, 32))))
    incident_day_of_week = float(st.sidebar.selectbox('Incident Day of Week', list(range(0, 7))))
    incident_week_of_year = float(st.sidebar.selectbox('Incident Week of Year', list(range(1, 53))))
    incident_quarter = float(st.sidebar.selectbox('Incident Quarter', list(range(1, 5))))

    # Label Encoding
    incident_severity = st.sidebar.selectbox('Incident Severity', ['Trivial Damage', 'Minor Damage', 'Major Damage', 'Total Loss'])
    incident_severity_encoded = label_encoder_severity.transform([incident_severity])[0]
    
    # One-Hot Encoding
    property_damage = st.sidebar.selectbox('Property Damage', ['NO', 'Unknown', 'YES'])
    police_report_available = st.sidebar.selectbox('Police Report Available', ['NO', 'Unknown', 'YES'])
    policy_state = st.sidebar.selectbox('Policy State', ['IL', 'IN', 'OH'])
    collision_type = st.sidebar.selectbox('Collision Type', ['Front Collision', 'Rear Collision', 'Side Collision', 'Unknown'])

    categorical_inputs = np.array([[property_damage, police_report_available, policy_state, collision_type]])
    categorical_inputs_encoded = one_hot_encoder.transform(categorical_inputs).toarray()

    # Scaling numeric features
    months_as_customer = float(st.sidebar.number_input('Months as Customer', min_value=0))
    total_claim_amount = float(st.sidebar.number_input('Total Claim Amount', min_value=0.0))
    policy_deductable = float(st.sidebar.number_input('Policy Deductable', min_value=0))
    injury_claim = float(st.sidebar.number_input('Injury Claim', min_value=0.0))
    property_claim = float(st.sidebar.number_input('Property Claim', min_value=0.0))
    vehicle_claim = float(st.sidebar.number_input('Vehicle Claim', min_value=0.0))
    age = float(st.sidebar.number_input('Age', min_value=0))

    numeric_inputs = np.array([[months_as_customer, total_claim_amount, policy_deductable, injury_claim, property_claim, vehicle_claim, age]])
    numeric_inputs_scaled = standard_scaler.transform(numeric_inputs)

    # Combine all processed features into a single array
    final_features = np.hstack((
        [incident_severity_encoded, incident_hour_of_the_day, number_of_vehicles_involved, bodily_injuries, witnesses,
         incident_year, incident_month, incident_day, incident_day_of_week, incident_week_of_year, incident_quarter],
        categorical_inputs_encoded.flatten(),  
        numeric_inputs_scaled.flatten()
    ))

    return final_features

# Main part of the app
input_data = user_input_features()

# Predict button
if st.button('Predict'):
    try:
        prediction = logistic_model.predict([input_data])
        
        if prediction[0] == 1:
            st.success('This claim is likely to  be dishonest or false.')
        else:
            st.success('This claim is likely to be true.')
    except ValueError as e:
        st.error(f"Error: {e}")

st.sidebar.markdown("""
**Note:** 
- Ensure all inputs are correctly filled.
- If you encounter issues, verify that the values align with the expected format.
""")
