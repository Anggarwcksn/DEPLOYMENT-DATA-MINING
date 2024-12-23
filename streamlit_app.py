import streamlit as st
import pandas as pd
import numpy as np
import base64
from pyngrok import ngrok

import streamlit as st
from pyngrok import ngrok, conf


# Load dataset untuk encoding lokasi
df = pd.read_csv('dataset.csv')  # Pastikan dataset.csv berada di folder yang sama
locations_encoded = pd.get_dummies(df['alamat'])  # One-Hot Encoding untuk lokasi

# Load model yang sudah dilatih
model = joblib.load('model_prediksi_harga_rumah.pkl')

# Custom CSS untuk background dan styling
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
        f"""
        <style>
            .stApp {{
                background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
                background-size: cover;
            }}
            
            .css-1d391kg {{
                background-color: rgba(255, 255, 255, 0.9);
                padding: 2rem;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }}
            
            .title {{
                text-align: center;
                color: #1E88E5;
                font-size: 2.5rem;
                font-weight: bold;
                margin-bottom: 2rem;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
            }}
            
            .prediction-box {{
                background-color: #1E88E5;
                color: white;
                padding: 1.5rem;
                border-radius: 10px;
                text-align: center;
                margin-top: 2rem;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }}
            
            .slider-label {{
                font-weight: bold;
                color: #1E88E5;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Add title
st.markdown('<p class="title">Prediksi Harga Rumah di Bandung</p>', unsafe_allow_html=True)

# Create columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.markdown('<p class="slider-label">Jumlah Kamar Tidur</p>', unsafe_allow_html=True)
    jk = st.slider('', 1, int(df['jk'].max()), 3)
    
    st.markdown('<p class="slider-label">Luas Bangunan (m²)</p>', unsafe_allow_html=True)
    lb = st.slider('', float(df['lb'].min()), float(df['lb'].max()), 500.0)

with col2:
    st.markdown('<p class="slider-label">Luas Tanah (m²)</p>', unsafe_allow_html=True)
    lt = st.slider('', float(df['lt'].min()), float(df['lt'].max()), 500.0)
    
    st.markdown('<p class="slider-label">Lokasi</p>', unsafe_allow_html=True)
    alamat = st.selectbox('', df['alamat'].unique())

# Prediction button
if st.button('Prediksi Harga', key='predict_button'):
    # One-Hot Encoding untuk lokasi berdasarkan dataset
    location_encoded = pd.DataFrame(columns=locations_encoded.columns)
    location_encoded.loc[0] = 0  # Isi semua kolom dengan nol
    location_encoded.loc[0, alamat] = 1  # Set kolom lokasi yang dipilih menjadi 1
    
    # Gabungkan fitur numerikal dan fitur lokasi
    numerical_features = np.array([[lt, lb, jk]])
    input_features = np.hstack([numerical_features, location_encoded.values])
    
    # Lakukan prediksi
    predicted_price = model.predict(input_features)
    
    # Tampilkan hasil prediksi
    st.markdown(
        f"""
        <div class="prediction-box">
            <h2>Hasil Prediksi</h2>
            <h3>Harga rumah impian Anda diperkirakan sekitar:</h3>
            <h1>IDR {predicted_price[0]:,.3f}.000</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

# Tambahkan background image
add_bg_from_local('background.jpg')  # Sesuaikan nama file background Anda
