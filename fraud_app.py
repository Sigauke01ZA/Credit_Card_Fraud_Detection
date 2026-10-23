import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

# Load the saved model and features
model = joblib.load(MODEL_DIR / "fraud_xgb_model.pkl")
features = joblib.load(MODEL_DIR / "fraud_features.pkl")

st.set_page_config(page_title="Fraud Detection Engine", layout="wide")
st.title("🛡️ Real-Time Credit Card Fraud Engine")
st.markdown("Simulate a transaction and see the XGBoost model's fraud probability in real-time.")

st.sidebar.header("Transaction Profile")
st.sidebar.markdown("*(Adjust the top drivers of fraud based on model feature importance)*")

# Top 5 Drivers for the Dashboard
amount = st.sidebar.number_input("Transaction Amount (R)", 1.0, 50000.0, 500.0)
deviation = st.sidebar.slider("Avg Amount Deviation (Sigma)", 0.0, 10.0, 1.0)
geo_velocity = st.sidebar.slider("Geo-Velocity (km/h)", 0.0, 2000.0, 10.0)
device_match = st.sidebar.selectbox("Device Fingerprint Match", ["Known Device", "Unknown Device", "New Device", "Spoofed"])
velocity_normal = st.sidebar.selectbox("Transaction Velocity (1h)", ["Within Normal", "Burst Pattern", "5x Baseline"])
cvv_status = st.sidebar.selectbox("CVV Match Status", ["Match", "Mismatch", "Not Provided"])

# ==============================================================
# THIS IS THE UPDATED SECTION WITH SAFE DEFAULTS
# ==============================================================

# Create a numeric DataFrame with all zeros
input_df = pd.DataFrame(0.0, index=[0], columns=features)

# Set realistic defaults for a "Normal" customer 
# (So the app doesn't assume a 0-second session is a bot)
input_df['session_duration_sec'] = 120.0    
input_df['account_age_days'] = 365.0        
input_df['failed_attempts_before_success'] = 0.0
input_df['chargeback_history_count'] = 0.0

# Apply the user's slider inputs
input_df['transaction_amount'] = amount
input_df['avg_amount_deviation_sigma'] = deviation
input_df['geo_velocity_kmh'] = geo_velocity

# One-Hot encoded columns mapping
if device_match == "Unknown Device":
    input_df['device_fingerprint_match_unknown_device'] = 1.0
elif device_match == "New Device":
    input_df['device_fingerprint_match_new_device'] = 1.0
elif device_match == "Spoofed":
    input_df['device_fingerprint_match_spoofed'] = 1.0

if velocity_normal == "Burst Pattern":
    input_df['transaction_velocity_1h_burst_pattern'] = 1.0
elif velocity_normal == "5x Baseline":
    input_df['transaction_velocity_1h_5x_baseline'] = 1.0

if cvv_status == "Mismatch":
    input_df['cvv_match_status_cvv2_mismatch'] = 1.0
elif cvv_status == "Not Provided":
    input_df['cvv_match_status_not_provided'] = 1.0

# ==============================================================
# END OF UPDATED SECTION
# ==============================================================

# Predict
fraud_prob = model.predict_proba(input_df)[0][1]

# Output
st.header("Transaction Risk Assessment")
col1, col2 = st.columns(2)

col1.metric("Fraud Probability", f"{fraud_prob:.2%}")
col2.metric("Risk Category", "HIGH RISK" if fraud_prob > 0.50 else "LOW RISK")

if fraud_prob > 0.50:
    st.error("🚨 **ACTION: BLOCK TRANSACTION.** This transaction exhibits strong patterns of fraud.")
elif fraud_prob > 0.20:
    st.warning("⚠️ **ACTION: MANUAL REVIEW.** Send an OTP (One-Time Password) or SMS verification to the customer.")
else:
    st.success("✅ **ACTION: APPROVE.** Transaction appears legitimate.")

# --- DEBUG / AUDIT SECTION ---
st.divider()
with st.expander("🔍 View Raw Model Inputs (Audit Trail)"):
    # Only show the columns that are greater than 0 to keep it clean
    active_features = input_df.loc[:, (input_df != 0).any(axis=0)]
    st.dataframe(active_features.T.rename(columns={0: "Value"}))