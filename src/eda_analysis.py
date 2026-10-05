import pandas as pd
import numpy as np
import os

def load_processed_data(file_path):
    print("1.Veri Yükleniyor...")
    df = pd.read_csv(file_path, encoding='utf-8')
    df.columns = df.columns.str.strip()
    print(f"Veri yüklendi: {df.shape[0]} satır, {df.shape[1]} sütun\n")
    return df

def feature_engineering(df):
    print("2. Özellik Mühendisliği Uygulanıyor...") 
    
    # Metin temizleme: Price ve kms_driven sütunlarındaki harf/virgül/özel karakterleri temizliyoruz
    if 'Price' in df.columns:
        # Metin olan 'Ask For Price' gibi değerleri ve virgülleri temizleme
        df['Price'] = df['Price'].astype(str).str.replace(',', '').str.extract(r'(\d+)')
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        
    if 'kms_driven' in df.columns:
        # 'kms', virgül vb. metinleri temizleyip sadece rakamları alma işlemleri
        df['kms_driven'] = df['kms_driven'].astype(str).str.replace(',', '').str.extract(r'(\d+)')
        df['kms_driven'] = pd.to_numeric(df['kms_driven'], errors='coerce')

    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df = df.dropna(subset=['year', 'Price', 'kms_driven'])
    df['car_age'] = 2026 - df['year']
    df = df[df['car_age'] >= 0]
    df['mileage_per_year'] = df['kms_driven'] / (df['car_age'] + 1)

    q1 = df['Price'].quantile(0.25)
    q3 = df['Price'].quantile(0.75)
    iqr = q3 - q1
    upper_limit = q3 + 1.5 * iqr
    lower_limit = q1 - 1.5 * iqr

    initial_count = df.shape[0]
    df = df[(df['Price'] >= lower_limit) & (df['Price'] <= upper_limit)]
    removed_count = initial_count - df.shape[0]
    
    print(f"Yeni özellikler eklendi.")
    print(f"Güncel veri boyutu: {df.shape[0]} satır, {df.shape[1]} sütun")
    print(f"Silinen satır sayısı: {removed_count}\n")
    return df

def save_engineered_data(df, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"veri seti kaydedildi: {output_path}\n")

if __name__ == "__main__":
    PROCESSED_DATA_PATH = "data/processed/cars_cleaned.csv"
    ENGINEERED_DATA_PATH = "data/processed/cars_featured.csv"
    
    df = load_processed_data(PROCESSED_DATA_PATH)
    df = feature_engineering(df)
    save_engineered_data(df, ENGINEERED_DATA_PATH)