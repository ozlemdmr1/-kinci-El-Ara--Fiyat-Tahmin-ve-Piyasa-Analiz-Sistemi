import pandas as pd
import os

def load_data(file_path):
    print("1. Ham Veri Yükleniyor...")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} bulunamadı.")
    df = pd.read_csv(file_path, sep=None, engine='python')
    print(f"Ham veri yüklendi: {df.shape[0]} satır, {df.shape[1]} sütun")
    return df

def clean_data(df):
    print("2. Veri Temizliği Başlatılıyor...")
    
    # Sütun adlarındaki boşlukları temizleme
    df.columns = df.columns.str.strip()
    
    # Mantıksız veya eksik verileri temizleme
    df = df.dropna()
    
    print(f"Temizlik tamamlandı: {df.shape[0]} geçerli satır kaldı.")
    return df

def save_data(df, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # index=False ve encoding='utf-8' ile dosyanın bozulmasını engelliyor.
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Temizlenmiş veri kaydedildi: {output_path}\n")

if __name__ == "__main__":
    RAW_DATA_PATH = "data/raw/cars_raw.csv"
    PROCESSED_DATA_PATH = "data/processed/cars_cleaned.csv"
    
    raw_df = load_data(RAW_DATA_PATH)
    cleaned_df = clean_data(raw_df)
    save_data(cleaned_df, PROCESSED_DATA_PATH)