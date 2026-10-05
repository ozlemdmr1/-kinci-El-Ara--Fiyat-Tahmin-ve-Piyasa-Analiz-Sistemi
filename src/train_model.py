import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def load_data(file_path):
    print("1. Optimize Edilmiş Veri Yükleniyor...")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} bulunamadı.")
    df = pd.read_csv(file_path, encoding='utf-8')
    print(f"Veri yüklendi: {df.shape[0]} satır, {df.shape[1]} sütun\n")
    return df

def prepare_data(df):
    print("2. Veri Hazırlığı ve Log Dönüşümü (np.log1p)...")
    
    X = df[['company', 'fuel_type', 'car_age', 'kms_driven', 'mileage_per_year']]
    
    # Log Dönüşümü: Fiyatı logaritma ölçeğine çekiyoruz
    y_log = np.log1p(df['Price'])
    
    X = pd.get_dummies(X, columns=['company', 'fuel_type'], drop_first=True)
    
    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )
    
    print(f"Eğitim Seti: {X_train.shape[0]} örnek | Test Seti: {X_test.shape[0]} örnek\n")
    return X_train, X_test, y_train_log, y_test_log, X.columns

def train_and_evaluate(X_train, X_test, y_train_log, y_test_log):
    print("3. Modeller Eğitiliyor ve Değerlendiriliyor...\n")
    
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    best_model = None
    best_r2 = -float('inf')
    best_name = ""
    
    # Gerçek fiyat ölçeğindeki hedef değerler
    y_test_actual = np.expm1(y_test_log)
    
    for name, model in models.items():
        model.fit(X_train, y_train_log)
        
        # Log ölçeğinde tahmin ve ardından GERÇEK TL ölçeğine geri çevirme (expm1)
        preds_log = model.predict(X_test)
        preds_actual = np.expm1(preds_log)
        
        # Gerçek değerler üzerinden metrikler
        r2 = r2_score(y_test_actual, preds_actual)
        rmse = np.sqrt(mean_squared_error(y_test_actual, preds_actual))
        
        print(f"--- {name} ---")
        print(f"  R² Score (Açıklayıcılık): {r2:.4f}")
        print(f"  RMSE (Ortalama Hata):     {rmse:.2f} TL\n")
        
        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_name = name
            
    print(f"En Başarılı Model: {best_name} (R² = {best_r2:.4f})\n")
    return best_model, best_name

def save_best_model(model, feature_names, model_dir="models"):
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "best_car_price_model.pkl"))
    joblib.dump(feature_names, os.path.join(model_dir, "model_features.pkl"))
    print(" Yenilenmiş ve iyileştirilmiş model diske kaydedildi!\n")

if __name__ == "__main__":
    DATA_PATH = "data/processed/cars_featured.csv"
    
    df = load_data(DATA_PATH)
    X_train, X_test, y_train_log, y_test_log, feature_names = prepare_data(df)
    best_model, best_name = train_and_evaluate(X_train, X_test, y_train_log, y_test_log)
    save_best_model(best_model, feature_names)