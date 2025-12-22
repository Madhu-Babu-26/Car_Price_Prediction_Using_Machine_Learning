

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st

# 1. Page Configuration (This shows in the browser tab)
st.set_page_config(
    page_title=" Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)

# 2. Centered Title using HTML/CSS
# This creates a <div> that aligns the text to the center
st.markdown("""
    <h1 style='text-align: center; color: #007bff; margin-top: -50px;'>
         Car  Price  Prediction
    </h1>
    <p style='text-align: center; font-size: 1.2em; color: #666;'>
    Instant AI-driven valuation for your vehicle
    </p>
    """, unsafe_allow_html=True)

# The rest of your car image and input code follows...
st.image("https://images.unsplash.com/photo-1492144534655-ae79c964c9d7?auto=format&fit=crop&q=80&w=1000", 
         use_container_width=True)

# 2. Load Assets from your Notebook
@st.cache_resource
def load_assets():
    model = joblib.load('car_price_rf_model.pkl')
    scaler = joblib.load('scaler.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    feature_cols = joblib.load('feature_columns.pkl')
    return model, scaler, label_encoders, feature_cols

try:
    model, scaler, encoders, feature_cols = load_assets()
except Exception as e:
    st.error(f"⚠️ Required files not found. Please ensure the .pkl files from your notebook are in the same folder.")
    st.stop()

# 3. Custom CSS for Professional Look (Fixed the Argument Error here)
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3em;
        background-color: #007bff;
        color: white;
        font-weight: bold;
    }
    .result-card {
        padding: 30px;
        background-color: white;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
        border: 1px solid #e0e0e0;
    }
    </style>
    """, unsafe_allow_html=True) # FIXED: Changed from unsafe_allow_usage to unsafe_allow_html

# # 4. Main Page Header & Image
# st.title("AutoValue: Smart Car Valuation")
# st.write("Enter the car details below to get an instant AI-driven price estimate.")

# # Display a professional car image
# st.image("https://images.unsplash.com/photo-1492144534655-ae79c964c9d7?auto=format&fit=crop&q=80&w=1000", width=1500)

# st.divider()

# 5. Main Page Inputs (No Sidebar)
st.subheader("🛠️ Vehicle Specifications")
col1, col2, col3 = st.columns(3)

with col1:
    name = st.selectbox("Brand & Model", encoders['name'].classes_)
    year = st.number_input("Year of Manufacture", 2000, 2025, 2018)
    km_driven = st.number_input("Total Kilometers Driven", 0, 1000000, 50000)

with col2:
    fuel = st.selectbox("Fuel Type", encoders['fuel'].classes_)
    transmission = st.selectbox("Transmission", encoders['transmission'].classes_)
    engine = st.number_input("Engine Capacity (CC)", 600, 8000, 1200)

with col3:
    seller_type = st.selectbox("Seller Type", encoders['seller_type'].classes_)
    owner = st.selectbox("Owner History", encoders['owner'].classes_)
    max_power = st.number_input("Max Power (bhp)", 30, 1000, 100)

# Optional Advanced Inputs
with st.expander("Additional Details"):
    c1, c2 = st.columns(2)
    with c1:
        seats = st.slider("Seats", 2, 10, 5)
        mileage = st.number_input("Mileage (kmpl)", 5.0, 50.0, 18.0)
    with c2:
        mileage_unit = st.selectbox("Mileage Unit", encoders['Mileage Unit'].classes_)

# 6. Prediction Logic
st.divider()
if st.button("Get Instant Valuation"):
    # Create Input DataFrame
    input_data = {
        'name': name, 'km_driven': km_driven, 'fuel': fuel,
        'seller_type': seller_type, 'transmission': transmission,
        'owner': owner, 'seats': seats, 'max_power (in bph)': max_power,
        'Mileage': mileage, 'Engine (CC)': engine, 'Mileage Unit': mileage_unit,
        'year': year
    }
    df_pred = pd.DataFrame([input_data])

    # Feature Engineering (Age, Price/KM, Power/Seat)
    df_pred['age'] = 2025 - df_pred['year']
    df_pred['price_per_km'] = 1.0 # Placeholder logic from your notebook
    df_pred['power_per_seat'] = df_pred['max_power (in bph)'] / df_pred['seats']

    # Encoding
    cat_cols = ['name', 'fuel', 'seller_type', 'transmission', 'owner', 'Mileage Unit']
    for col in cat_cols:
        df_pred[col] = encoders[col].transform(df_pred[col])

    # Final Prediction Process
    X_pred = df_pred[feature_cols]
    X_pred_scaled = scaler.transform(X_pred)
    prediction = model.predict(X_pred_scaled)[0]

    # Display Result
    st.markdown(f"""
        <div class="result-card">
            <h3 style='color: #666;'>Estimated Selling Price</h3>
            <h1 style='color: #28a745; font-size: 50px;'>₹ {prediction:,.2f}</h1>
        </div>
    """, unsafe_allow_html=True)
    st.balloons()
