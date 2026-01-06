import pandas as pd
import numpy as np
import joblib
import re
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# --- Helper 1: Extract Horsepower ---
def extract_hp(engine_str):
    try:
        if isinstance(engine_str, str):
            match = re.search(r'(\d+)\s*HP', engine_str, re.IGNORECASE)
            if match:
                return float(match.group(1))
    except:
        pass
    return None

# --- Helper 2: Clean Mileage ---
def clean_mileage(milage_val):
    try:
        if isinstance(milage_val, (int, float)):
            return float(milage_val)
        if isinstance(milage_val, str):
            clean_str = re.sub(r'[^\d]', '', milage_val) 
            if clean_str:
                return float(clean_str)
    except:
        pass
    return None

# 1. Load Data
try:
    pd.read_csv("data/used_cars.csv")

    df.columns = df.columns.str.lower()
    print(f"✅ Data loaded! Columns: {list(df.columns)}")
except FileNotFoundError:
    print("❌ Error: 'used_cars.csv' not found.")
    exit()

# 2. FEATURE ENGINEERING (The "Smart" Fixes)

# A. Handle Age
if 'model_year' in df.columns:
    df['car_age'] = 2025 - df['model_year']
else:
    df['car_age'] = 5

# B. Handle Horsepower
if 'engine' in df.columns:
    print("⚙️ Cleaning 'engine' column...")
    df['horsepower'] = df['engine'].apply(extract_hp).fillna(250.0)
else:
    df['horsepower'] = 250.0

# C. Handle Mileage
print("⚙️ Cleaning 'milage' column...")
# Handle spelling variation
if 'milage' not in df.columns and 'mileage' in df.columns:
    df['milage'] = df['mileage']
elif 'milage' not in df.columns:
    df['milage'] = 50000 # Default
df['milage'] = df['milage'].apply(clean_mileage)

# D. SMART MODEL GROUPING (The Critical Fix)
if 'model' in df.columns:
    # Get the top 20 most frequent models (e.g., F-150, Civic, Accord)
    top_models = df['model'].value_counts().nlargest(20).index.tolist()
    print(f"🧠 Smart Grouping: Keeping these top models specific: {top_models}")
    
    # If a car is in the top 20, keep its name. Otherwise, mark it 'Other'.
    df['model_grouped'] = df['model'].apply(lambda x: x if x in top_models else 'Other')
else:
    df['model_grouped'] = 'Other'

# E. Handle other missing defaults
required = {'is_performance': 0, 'accident': 'None reported', 'clean_title': 'Yes'}
for col, val in required.items():
    if col not in df.columns:
        df[col] = val

# 3. Setup Training
features = ['brand', 'model_grouped', 'milage', 'car_age', 'horsepower', 
            'is_performance', 'fuel_type', 'transmission', 'accident', 'clean_title']
target = 'price'

# Drop bad rows
df = df.dropna(subset=[target, 'car_age', 'horsepower', 'milage'])

# Clean Price Column if needed
if df[target].dtype == object:
     df[target] = df[target].astype(str).str.replace(r'[^\d]', '', regex=True).astype(float)

X = df[features]
y = df[target]

# 4. Build Pipeline
num_feats = ['milage', 'car_age', 'horsepower', 'is_performance']
cat_feats = ['brand', 'model_grouped', 'fuel_type', 'transmission', 'accident', 'clean_title']

preprocessor = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_feats),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat_feats)
])

model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=50, random_state=42))
])

# 5. Train and Save
print("⏳ Training model with Smart Grouping...")
model.fit(X, y)
joblib.dump(model, 'best_car_model.joblib')
print("🎉 Success! Smarter model saved as 'best_car_model.joblib'.")
