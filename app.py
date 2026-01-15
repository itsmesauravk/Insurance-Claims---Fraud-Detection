import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- Streamlit Configuration ---
st.set_page_config(
    page_title="Insurance Claims Fraud Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Insurance Claims Fraud Detection")
st.markdown("""
This site uses Random Forest Classifier to predict fraudulent insurance claims. Use the sidebar to input details for claim prediction.
""")

# --- Constants and Caching Functions for Efficiency ---

FILE_PATH = os.getenv("DATA_PATH", "csv/fraud_insurance_claims.csv")

@st.cache_data
def load_data(file_path):
    """Loads the raw data."""
    # Note: Streamlit runs the script from its location, so the path is relative.
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"File not found: {file_path}. Please ensure it is available.")
        st.stop()
    return df

@st.cache_data
def preprocess_data(df):
    """
    Performs the necessary preprocessing steps (dropping columns,
    target encoding, and one-hot encoding).
    """
    df_copy = df.copy()
    
    # 1. Target Encoding (Y -> 1, N -> 0, and ensure integer type)
    df_copy['fraud_reported'] = df_copy['fraud_reported'].map({'N': 0, 'Y': 1}).astype(int)

    # 2. Drop columns that are irrelevant or problematic for direct modeling
    cols_to_drop = [
        'policy_bind_date', 'incident_date', 'policy_number', 
        'insured_zip', 'incident_location', 'fraud_flag', 'reviews',
        'auto_model' # High cardinality, removed for simpler model
    ]
    df_cleaned = df_copy.drop(columns=cols_to_drop, errors='ignore')
    
    # Extract original categorical columns for mapping
    categorical_cols = df_cleaned.select_dtypes(include='object').columns.tolist()

    # 3. One-hot encode remaining categorical columns ('object' dtype)
    df_fe = pd.get_dummies(df_cleaned, columns=categorical_cols, drop_first=True)
    
    # Define X and y
    X = df_fe.drop(columns=["fraud_reported"])
    y = df_fe["fraud_reported"]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.30, random_state=42
    )
    return X_train, X_test, y_train, y_test, X, y, categorical_cols


@st.cache_resource
def train_random_forest(X_train, y_train):
    """Trains the Random Forest Classifier with specified hyperparameters."""
    with st.spinner('Training Random Forest Model...'):
        rfc = RandomForestClassifier(
            criterion='entropy', 
            max_depth=12, 
            n_estimators=300, 
            random_state=42
        )
        rfc.fit(X_train, y_train)
    return rfc

# --- Load Data and Train Model ---

df_raw = load_data(FILE_PATH)
X_train, X_test, y_train, y_test, X_full, y_full, categorical_cols = preprocess_data(df_raw)
rfc = train_random_forest(X_train, y_train)

# --- Define Available Choices for User Inputs ---
CHOICES = {}
for col in categorical_cols:
    # Use unique values from the raw data for selection boxes
    CHOICES[col] = df_raw[col].dropna().unique().tolist()
    
# Filter out numeric columns we don't want input for (like claim amounts which are input separately)
NUMERIC_FEATURES_FOR_INPUT = [
    'months_as_customer', 'age', 'policy_deductable', 
    'total_claim_amount', 'injury_claim', 'property_claim', 'vehicle_claim',
    'incident_hour_of_the_day', 'number_of_vehicles_involved', 
    'body_count', 'witnesses', 'auto_year'
]


# --- Prediction Form in Sidebar ---

st.sidebar.header("Claim Details for Prediction")
st.sidebar.markdown("Enter all details to generate a fraud prediction.")

# --- Helper function for collecting all inputs ---
def user_input_features():
    # 1. Numeric Inputs (from original data columns)
    st.sidebar.subheader("Financial & Demographic Details")
    months_as_customer = st.sidebar.number_input('Months as Customer', min_value=0, value=int(df_raw['months_as_customer'].mean()), step=1)
    age = st.sidebar.number_input('Age', min_value=18, max_value=100, value=int(df_raw['age'].mean()), step=1)
    policy_deductable = st.sidebar.number_input('Policy Deductable', min_value=0, max_value=5000, value=int(df_raw['policy_deductable'].mean()), step=100)
    
    st.sidebar.subheader("Incident Details")
    incident_hour_of_the_day = st.sidebar.slider('Incident Hour of the Day', 0, 23, int(df_raw['incident_hour_of_the_day'].mean()))
    number_of_vehicles_involved = st.sidebar.slider('Number of Vehicles Involved', 1, 4, 1)
    body_count = st.sidebar.slider('Body Count', 0, 3, 1)
    witnesses = st.sidebar.slider('Witnesses', 0, 5, 1)
    auto_year = st.sidebar.slider('Auto Year', 1990, 2020, 2010)

    st.sidebar.subheader("Claim Amounts")
    total_claim_amount = st.sidebar.number_input('Total Claim Amount', min_value=0.0, value=float(df_raw['total_claim_amount'].mean()), step=500.0)
    injury_claim = st.sidebar.number_input('Injury Claim', min_value=0.0, value=float(df_raw['injury_claim'].mean()), step=100.0)
    property_claim = st.sidebar.number_input('Property Claim', min_value=0.0, value=float(df_raw['property_claim'].mean()), step=100.0)
    vehicle_claim = st.sidebar.number_input('Vehicle Claim', min_value=0.0, value=float(df_raw['vehicle_claim'].mean()), step=500.0)
    
    st.sidebar.subheader("Categorical Features")
    # Categorical Inputs
    insured_sex = st.sidebar.selectbox('Insured Sex', options=CHOICES['insured_sex'])
    insured_education_level = st.sidebar.selectbox('Insured Education Level', options=CHOICES['insured_education_level'])
    insured_occupation = st.sidebar.selectbox('Insured Occupation', options=CHOICES['insured_occupation'])
    insured_hobbies = st.sidebar.selectbox('Insured Hobbies', options=CHOICES['insured_hobbies'])
    insured_relationship = st.sidebar.selectbox('Insured Relationship', options=CHOICES['insured_relationship'])
    incident_type = st.sidebar.selectbox('Incident Type', options=CHOICES['incident_type'])
    collision_type = st.sidebar.selectbox('Collision Type', options=CHOICES['collision_type'])
    incident_severity = st.sidebar.selectbox('Incident Severity', options=CHOICES['incident_severity'])
    authorities_contacted = st.sidebar.selectbox('Authorities Contacted', options=CHOICES['authorities_contacted'])
    incident_state = st.sidebar.selectbox('Incident State', options=CHOICES['incident_state'])
    incident_city = st.sidebar.selectbox('Incident City', options=CHOICES['incident_city'])
    property_damage = st.sidebar.selectbox('Property Damage', options=CHOICES['property_damage'])
    police_report_available = st.sidebar.selectbox('Police Report Available', options=CHOICES['police_report_available'])
    auto_make = st.sidebar.selectbox('Auto Make', options=CHOICES['auto_make'])
    
    # The 'policy_state' is not in the original drop list, but let's include it for completeness
    # Based on the original notebook, let's include policy_state
    policy_state = st.sidebar.selectbox('Policy State', options=CHOICES['policy_state'])


    data = {
        'months_as_customer': months_as_customer,
        'age': age,
        'policy_csl': df_raw['policy_csl'].mode()[0], # Use mode for simplicity as it's not exposed
        'policy_deductable': policy_deductable,
        'policy_annual_premium': df_raw['policy_annual_premium'].mean(), # Use mean for simplicity
        'umbrella_limit': df_raw['umbrella_limit'].mode()[0], # Use mode for simplicity
        'capital-gains': df_raw['capital-gains'].mean(), # Use mean for simplicity
        'capital-loss': df_raw['capital-loss'].mean(), # Use mean for simplicity
        'incident_hour_of_the_day': incident_hour_of_the_day,
        'number_of_vehicles_involved': number_of_vehicles_involved,
        'body_count': body_count,
        'witnesses': witnesses,
        'total_claim_amount': total_claim_amount,
        'injury_claim': injury_claim,
        'property_claim': property_claim,
        'vehicle_claim': vehicle_claim,
        'auto_year': auto_year,
        'insured_sex': insured_sex,
        'insured_education_level': insured_education_level,
        'insured_occupation': insured_occupation,
        'insured_hobbies': insured_hobbies,
        'insured_relationship': insured_relationship,
        'incident_type': incident_type,
        'collision_type': collision_type,
        'incident_severity': incident_severity,
        'authorities_contacted': authorities_contacted,
        'incident_state': incident_state,
        'incident_city': incident_city,
        'property_damage': property_damage,
        'police_report_available': police_report_available,
        'auto_make': auto_make,
        'policy_state': policy_state
    }
    
    return data

# --- Prediction Logic ---

def predict_single_claim(model, X_train_cols, input_data):
    # 1. Convert input data dictionary into a DataFrame with one row
    input_df = pd.DataFrame([input_data])
    
    # 2. Select columns that are present in the model's categorical list
    cols_for_dummies = [col for col in categorical_cols if col in input_df.columns]
    
    # 3. Perform One-Hot Encoding on the new input data (must match training data structure)
    # The columns must exactly match the ones used in preprocess_data()
    
    # A. Create a DataFrame for categorical features ONLY
    cat_df = input_df[cols_for_dummies]
    
    # B. Apply pd.get_dummies, ensuring we only use levels seen during training (stored in X_full.columns)
    # This creates the one-hot encoded columns for the input row.
    input_encoded_df = pd.get_dummies(cat_df, columns=cols_for_dummies, drop_first=True)
    
    # C. Drop the original categorical columns and append the new encoded columns to the numeric features
    input_final_df = input_df.drop(columns=cols_for_dummies, errors='ignore').join(input_encoded_df)
    
    # 4. Align the columns with the training data (X_full.columns)
    # Create the final prediction dataframe initialized to zero for all model features
    final_prediction_df = pd.DataFrame(0, index=[0], columns=X_full.columns)
    
    # Copy the values from the input row into the correctly structured frame
    for col in final_prediction_df.columns:
        if col in input_final_df.columns:
            final_prediction_df.loc[0, col] = input_final_df.loc[0, col]
            
    # 5. Predict
    prediction_proba = model.predict_proba(final_prediction_df)[0]
    prediction = model.predict(final_prediction_df)[0]
    
    return prediction, prediction_proba

# Main part of the app
input_data = user_input_features()

# Predict button
st.sidebar.markdown("---")
if st.sidebar.button('Predict Fraud Likelihood', type="primary"):
    
    prediction, proba = predict_single_claim(
        rfc, 
        X_full.columns, 
        input_data
    )
    
    st.header("Prediction Result")
    
    if prediction == 1:
        st.error(f"🔴 **HIGH FRAUD RISK**")
        st.markdown(f"The model predicts this claim is **FRAUDULENT** with **{proba[1]*100:.2f}%** confidence.")
    else:
        st.success(f"🟢 **LOW FRAUD RISK**")
        st.markdown(f"The model predicts this claim is **NOT FRAUDULENT** with **{proba[0]*100:.2f}%** confidence.")
        
    st.progress(proba[1], text="Fraud Probability")

st.header("Data Overview")
st.dataframe(X_full.head())