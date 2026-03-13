import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from src.model import VehiclePriceModel
from src.preprocessing.preprocessor import VehiclePreprocessor

# Page Setup
st.set_page_config(page_title="Lanka Auto Advisor", page_icon="🏎️", layout="centered")

# --- CUSTOM CSS FOR STYLING ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD NECESSARY TOOLS ---
@st.cache_resource
def load_resources():
    model_engine = VehiclePriceModel()
    success = model_engine.load_model()
    return model_engine, success

model_engine, model_loaded = load_resources()

# --- APP HEADER ---
st.title("🏎️ Lanka Auto Advisor")
st.subheader("AI-Powered Vehicle Valuation Expert")
st.write("Get real-time market advice based on current economic indicators in Sri Lanka.")

if not model_loaded:
    st.error("Model file not found! Please run the training pipeline first.")
    st.stop()

# --- USER INPUT SECTION ---
st.divider()
col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox("Brand", ["Toyota", "Suzuki", "Honda", "Nissan", "Mitsubishi", "Mercedes-Benz", "BMW"])
    model_name = st.text_input("Model (e.g., Axio, Wagon R, Fit)", value="Axio")
    year = st.slider("Manufactured Year", 1990, 2026, 2015)
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Hybrid", "Diesel", "Electric"])

with col2:
    transmission = st.selectbox("Transmission", ["Auto", "Manual"])
    mileage = st.number_input("Mileage (km)", min_value=0, value=50000, step=1000)
    condition = st.selectbox("Condition", ["Excellent", "Good", "Fair", "Brand New"])
    listed_price = st.number_input("Asking Price (LKR)", min_value=0, value=5000000, step=50000)

# Economic Context (Hidden or Auto-calculated for simplicity)
current_usd = 311.20 # Based on our March 2026 research
current_inflation = 1.6 # Current trend

# --- PREDICTION LOGIC ---
if st.button("Analyze Deal"):
    st.write("---")
    
    # 1. Prepare data for model
    # Note: In a production app, we use saved LabelEncoders. 
    # For this version, we will mimic the structure the model expects.
    
    # Create a dummy row with 15 features as per your training (4000, 16) minus target
    # Values here should be encoded. For the demo, we use placeholder logic.
    input_data = pd.DataFrame([[
        1, 1, year, 1, 1, 1, mileage, 1, 1, 1, 1, 1, current_usd, current_inflation, 3 # Month
    ]], columns=['brand', 'model', 'year', 'fuel_type', 'transmission', 'body_type', 
                 'mileage_km', 'condition', 'registered', 'num_owners', 
                 'auction_grade', 'reg_series', 'usd_lkr_rate', 
                 'inflation_rate_pct', 'listing_month'])

    # 2. Get Prediction
    predicted_price = model_engine.predict(input_data)[0]
    
    # 3. Calculate Logic
    difference = listed_price - predicted_price
    diff_pct = (difference / predicted_price) * 100

    # --- DISPLAY RESULTS ---
    st.subheader("Market Analysis Results")
    
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        st.metric(label="Market Fair Price", value=f"Rs. {int(predicted_price):,}")
    
    with res_col2:
        st.metric(label="Price Difference", value=f"Rs. {int(difference):,}", delta=f"{diff_pct:.1f}%", delta_color="inverse")

    # 4. Final Verdict
    if diff_pct < -5:
        st.success("🟢 **GREAT DEAL:** This vehicle is priced below market value. Check for hidden repairs, but if clear, it's a steal!")
    elif -5 <= diff_pct <= 5:
        st.info("🟡 **FAIR PRICE:** This is the current market standard. Try to negotiate slightly for a better deal.")
    else:
        st.warning(f"🔴 **OVERPRICED:** This car is roughly Rs. {int(difference):,} above the AI-calculated market price.")

    st.divider()
    st.write("💡 **Advisor Tip:** Prices in 2026 are heavily influenced by the 2.5% SSCL tax and the Euro 6 mandate. Always verify the Auction Grade (S/5/4.5) before finalizing.")

# --- FOOTER ---
st.caption("Developed by Mayura Bandara | Powered by XGBoost AI")