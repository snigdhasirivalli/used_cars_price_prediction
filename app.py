import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Car Price AI", page_icon="🚗")
st.title("🚗 Used Car Price Predictor")

# --- Load Model ---
@st.cache_resource
def load_model():
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(curr_dir, 'best_car_model.joblib')
    return joblib.load(model_path)

try:
    model = load_model()
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# --- User Inputs ---
st.sidebar.header("Enter Car Details")

# 1. Brands
brands = ['Ford', 'Hyundai', 'Lexus', 'INFINITI', 'Audi', 'Acura', 'BMW', 'Mercedes-Benz', 
          'Toyota', 'Honda', 'Chevrolet', 'Nissan', 'Jeep', 'Volkswagen', 'Subaru', 
          'Kia', 'Dodge', 'GMC', 'Mazda', 'Ram']
brand = st.sidebar.selectbox("Brand", sorted(brands))

# 2. Car Models (The "Unlocked" Logic)
# These are the top models your AI was trained on. 
# NOTE: If your 'train.py' printed a slightly different list, update this!
top_models = ['F-150', 'Silverado 1500', 'Ram 1500', 'Civic', 'Accord', 
              'Altima', 'Escape', 'Camry', 'Fusion', 'CR-V', 
              'Explorer', 'Equinox', 'Grand Cherokee', 'Corolla', 
              'Mustang', 'Malibu', 'Wrangler', 'Sierra 1500', 'Elantra', 'Sentra']

# Add 'Other' for cars not in the top 20
model_options = ['Other'] + sorted(top_models)
selected_model = st.sidebar.selectbox("Car Model (Top 20)", model_options)

# 3. Other Details
year = st.sidebar.number_input("Year", 1990, 2025, 2019)
mileage = st.sidebar.number_input("Mileage", 0, 300000, 50000)
horsepower = st.sidebar.number_input("Horsepower", 50, 1000, 200)

is_perf = st.sidebar.radio("Performance Trim?", ("No", "Yes"))
is_perf_val = 1 if is_perf == "Yes" else 0

fuel = st.sidebar.selectbox("Fuel", ['Gasoline', 'Hybrid', 'Diesel', 'E85 Flex Fuel'])
trans = st.sidebar.selectbox("Transmission", ['Automatic', 'A/T', 'Manual', 'CVT'])

# --- Prediction Logic ---
if st.button("Predict Price"):
    age = 2025 - year
    
    # We now pass the ACTUAL selected model (e.g., 'F-150') instead of always 'Other'
    input_data = pd.DataFrame({
        'brand': [brand], 
        'model_grouped': [selected_model],  # <--- The Critical Fix
        'milage': [mileage],           
        'car_age': [age], 
        'horsepower': [horsepower], 
        'is_performance': [is_perf_val],
        'fuel_type': [fuel], 
        'transmission': [trans],
        'accident': ['None reported'], 
        'clean_title': ['Yes']         
    })
    
    try:
        pred = model.predict(input_data)[0]
        st.success(f"💰 Estimated Price: ${pred:,.2f}")
    except Exception as e:
        st.error(f"Prediction Error: {e}")