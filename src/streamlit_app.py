import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Sayfa Yapılandırması
st.set_page_config(
    page_title="İkinci El Araç Fiyat Tahmin Sistemi",
    page_icon="🚗",
    layout="centered"
)

# Model ve Özellik Yollarını Belirleme
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_car_price_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "models", "model_features.pkl")

@st.cache_resource
def load_ml_components():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
        return None, None
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
    return model, features

model, model_features = load_ml_components()

# Başlık ve Açıklama
st.title("İkinci El Araç Fiyat Tahmin Sistemi")
st.markdown("Eğitilmiş Makine Öğrenmesi (Random Forest) modeli ile aracınızın piyasa değerini anında hesaplayın.")
st.divider()

if model is None:
    st.error("Model dosyası bulunamadı! Lütfen önce `train_model.py` script'ini çalıştırın.")
else:
    # Form Alanları
    st.subheader("Araç Bilgilerini Girin")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Marka Seçenekleri (Veri setindeki yaygın markalar)
        company = st.selectbox(
            "Araç Markası",
            ["Maruti", "Hyundai", "Mahindra", "Tata", "Honda", "Toyota", "Ford", "Chevrolet", "Renault", "Volkswagen", "Audi", "BMW"]
        )
        year = st.slider("Model Yılı", min_value=2000, max_value=2026, value=2018)

    with col2:
        fuel_type = st.selectbox("Yakıt Tipi", ["Petrol", "Diesel", "LPG", "CNG"])
        kms_driven = st.number_input("Kilometre (KM)", min_value=0, max_value=500000, value=45000, step=1000)

    st.divider()

    # Tahmin Butonu
    if st.button("Fiyat Tahmini Yap", type="primary", use_container_width=True):
        # Feature Engineering hesaplamaları
        current_year = 2026
        car_age = current_year - year
        if car_age < 0:
            car_age = 0
            
        mileage_per_year = kms_driven / (car_age + 1)

        input_data = {
            'company': company,
            'fuel_type': fuel_type,
            'car_age': car_age,
            'kms_driven': kms_driven,
            'mileage_per_year': mileage_per_year
        }

        # DataFrame ve One-Hot Encoding
        df_input = pd.DataFrame([input_data])
        df_encoded = pd.get_dummies(df_input, columns=['company', 'fuel_type'])
        df_final = df_encoded.reindex(columns=model_features, fill_value=0)

        # Tahmin (Log -> Gerçek TL)
        log_pred = model.predict(df_final)[0]
        actual_pred = float(np.expm1(log_pred))

        # Sonuç Gösterimi
        st.balloons()
        st.success(f"###Tahmini Araç Fiyatı: **{round(actual_pred, 2):,} TL**".replace(",", "."))