import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# ----------------------------------
# PAGE CONFIG
# ----------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="wide"
)

st.title("Customer Churn Prediction with Explainable AI")

# ----------------------------------
# LOAD MODEL & ENCODERS
# ----------------------------------
@st.cache_resource
def load_assets():
    model = joblib.load("Customer_churn_model.pkl")
    encoders = joblib.load("label_encoders.pkl")
    return model, encoders

model, encoders = load_assets()

# ----------------------------------
# LOAD & PREPROCESS DATASET
# ----------------------------------
# ----------------------------------
# LOAD DATASET & ENCODE IMMEDIATELY
# ----------------------------------

df = pd.read_csv("Churn_Modelling.xls") 


# 1. Clean out non-numeric structural columns if they exist in your file
drop_cols = ["RowNumber", "CustomerId", "Surname", "Exited"]
X = df.drop(columns=[col for col in drop_cols if col in df.columns])

# 2. Encode the text columns so XGBoost receives integers, not 'object' strings
X["Geography"] = encoders["Geography"].transform(X["Geography"])
X["Gender"] = encoders["Gender"].transform(X["Gender"])

# Ensure all columns are numeric data types for XGBoost safety
X["Geography"] = X["Geography"].astype(int)
X["Gender"] = X["Gender"].astype(int)


# ----------------------------------
# SHAP EXPLAINER (Enforce probability output if supported)
# ----------------------------------
@st.cache_resource
def get_explainer(_model, _X_sample):
    try:
        # Request probability scale interpretations if tree-based model
        return shap.TreeExplainer(_model, model_output="probability")
    except Exception:
        return shap.TreeExplainer(_model)

explainer = get_explainer(model, X.head(100))

# ----------------------------------
# SIDEBAR MODE SELECTOR
# ----------------------------------
mode = st.sidebar.radio(
    "Select Mode",
    ["Existing Customer", "New Customer"]
)

# ==================================
# EXISTING CUSTOMER MODE
# ==================================
if mode == "Existing Customer":
    customer_id = st.selectbox(
        "Select Customer Row Index",
        X.index
    )

    # Slice single customer row
    customer = X.iloc[[customer_id]]

    st.subheader("Customer Details")
    st.dataframe(customer)

    # Inference predictions
    pred = model.predict(customer)[0]
    prob = model.predict_proba(customer)[0][1]

    st.subheader("Prediction")
    st.metric("Churn Probability", f"{prob:.2%}")

    if pred == 1:
        st.error("Likely To Churn")
    else:
        st.success("Likely To Stay")

    # SHAP Waterfall Plot
    st.subheader("Why was this prediction made?")
    shap_values = explainer(customer)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)

# ==================================
# NEW CUSTOMER MODE
# ==================================
else:
    st.subheader("Enter Customer Information")

    col1, col2 = st.columns(2)

    with col1:
        credit_score = st.number_input("Credit Score", 300, 900, 650)
        geography = st.selectbox("Geography", encoders["Geography"].classes_)
        gender = st.selectbox("Gender", encoders["Gender"].classes_)
        age = st.number_input("Age", 18, 100, 35)
        tenure = st.number_input("Tenure", 0, 20, 5)

    with col2:
        balance = st.number_input("Balance", value=50000.0)
        num_products = st.number_input("Number of Products", 1, 4, 1)
        has_card = st.selectbox("Has Credit Card (0=No, 1=Yes)", [0, 1])
        active_member = st.selectbox("Is Active Member (0=No, 1=Yes)", [0, 1])
        salary = st.number_input("Estimated Salary", value=50000.0)

    if st.button("Predict Churn", type="primary"):
        
        # 1. Properly Encode categorical user inputs using your loaded LabelEncoders
        geography_encoded = encoders["Geography"].transform([geography])[0]
        gender_encoded = encoders["Gender"].transform([gender])[0]

        # 2. Build DataFrame ensuring features follow the exact training order of X
        customer_new = pd.DataFrame([{
            "CreditScore": credit_score,
            "Geography": geography_encoded,
            "Gender": gender_encoded,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": num_products,
            "HasCrCard": has_card,
            "IsActiveMember": active_member,
            "EstimatedSalary": salary
        }], columns=X.columns) # Force column sorting order match

        # 3. Model Inference predictions
        pred = model.predict(customer_new)[0]
        prob = model.predict_proba(customer_new)[0][1]

        st.subheader("Prediction")
        st.metric("Churn Probability", f"{prob:.2%}")

        if pred == 1:
            st.error("Likely To Churn")
        else:
            st.success("Likely To Stay")

        # 4. Generate SHAP Explanation
        st.subheader("Prediction Explanation")
        shap_values_new = explainer(customer_new)

        fig, ax = plt.subplots(figsize=(10, 5))
        shap.plots.waterfall(shap_values_new[0], show=False)
        st.pyplot(fig)

# ==================================
# GLOBAL EXPLANATION SECTION
# ==================================
st.divider()
st.header("Global Model Explanation")

if st.button("Show SHAP Summary Plot"):
    with st.spinner("Calculating overall feature importance... This might take a second."):
        # Sample down to 500 rows if your base dataset is massive to prevent out-of-memory errors
        X_sample = X.sample(min(500, len(X)), random_state=42)
        shap_values_global = explainer(X_sample)

        fig, ax = plt.subplots(figsize=(10, 6))
        # Note: target a beeswarm plot using modern SHAP API signatures
        shap.plots.beeswarm(shap_values_global, show=False)
        st.pyplot(fig)
