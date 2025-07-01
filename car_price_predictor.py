import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Set page config (must be FIRST)
st.set_page_config(page_title="🚗 Car Price Predictor", layout="centered")

# Custom dark theme CSS
st.markdown("""
    <style>
        .stApp {
            background-color: #0f0f0f;
        }
        h1, h2, h3, h4, h5, h6, p, div, span {
            color: white !important;
        }
        .block-container {
            padding: 2rem 1rem;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🚗 Car Price Predictor")
st.markdown("Enter the car's features to get an estimated price")

# Load and clean data
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\PARTH\Documents\Summer internship\Streamlit_all_project\car_data.csv")  # Adjust path as needed
    df['Price'] = df['Price'].str.replace("Rs. ", "").str.replace(" Lakh", "").str.replace(",", "")
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce') * 100000

    df['Rating'] = df['Rating'].str.replace("/5", "").astype(float)
    df['Safety'] = df['Safety'].str.extract(r"(\d+)").astype(float)
    df['Mileage'] = df['Mileage'].str.extract(r"(\d+\.?\d*)").astype(float)
    df['Power (BHP)'] = df['Power (BHP)'].str.extract(r"(\d+\.?\d*)").astype(float)

    df.dropna(subset=['Price', 'Rating', 'Safety', 'Mileage', 'Power (BHP)'], inplace=True)
    return df

df = load_data()
st.success("✅ Data Loaded and Cleaned Successfully")

# Sidebar Inputs
st.sidebar.header("📊 Input Car Details")

brands = sorted(df['Brand'].dropna().unique())
car_names = sorted(df['Car Name'].dropna().unique())

brand_input = st.sidebar.selectbox("🏷️ Brand", brands)
model_input = st.sidebar.selectbox("🚘 Car Model", car_names)

rating = st.sidebar.slider("⭐ Rating", 3.0, 5.0, 4.5, step=0.1)
safety = st.sidebar.slider("🛡️ Safety (Stars)", 1.0, 5.0, 4.0, step=1.0)
mileage = st.sidebar.slider("⛽ Mileage (kmpl)", 5.0, 40.0, 18.0, step=0.5)
power = st.sidebar.slider("⚡ Power (BHP)", 50.0, 850.0, 100.0, step=10.0)

# Train model
X = df[['Rating', 'Safety', 'Mileage', 'Power (BHP)']]
y = df['Price']
model = LinearRegression()
model.fit(X, y)

# Prediction
input_data = np.array([[rating, safety, mileage, power]])
predicted_price = model.predict(input_data)[0]

# Show prediction
st.markdown("### 🧾 **Prediction Result**")
st.markdown(f"""
<div style="background-color:#1e1e1e;padding:20px;border-radius:15px;box-shadow:0 2px 5px rgba(255,255,255,0.1);">
    <h3 style="color:#00e676;">💰 Estimated Price: ₹ {predicted_price:,.0f}</h3>
    <ul style="line-height:2em;color:white;">
        <li><b>🏷️ Brand:</b> {brand_input}</li>
        <li><b>🚘 Car Model:</b> {model_input}</li>
        <li><b>⭐ Rating:</b> {rating}</li>
        <li><b>🛡️ Safety:</b> {safety} Stars</li>
        <li><b>⛽ Mileage:</b> {mileage} kmpl</li>
        <li><b>⚡ Power:</b> {power} BHP</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Optional: Show full cleaned data
with st.expander("📄 View Cleaned Dataset"):
    st.dataframe(df)
