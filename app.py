import streamlit as st
import pandas as pd
import joblib
import os

st.title("Prediksi Penyewaan Sepeda")

# ========================
# Load Model dengan aman
# ========================
BASE_DIR = os.path.dirname(__file__)

model_path = os.path.join(BASE_DIR, 'model_xgb.pkl')
columns_path = os.path.join(BASE_DIR, 'columns.pkl')

# Debug (biar kamu sadar isi folder apa)
st.write("Isi folder:", os.listdir(BASE_DIR))

# Cek file
if not os.path.exists(model_path):
    st.error("model_xgb.pkl tidak ditemukan. Upload dulu ke project.")
    st.stop()

if not os.path.exists(columns_path):
    st.error("columns.pkl tidak ditemukan. Upload dulu ke project.")
    st.stop()

# Load
model = joblib.load(model_path)
training_columns = joblib.load(columns_path)

# ========================
# Upload file CSV
# ========================
uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Data Input:", df.head())

    # ========================
    # Preprocessing
    # ========================
    categorical_cols = [
        'season', 'yr', 'mnth', 'holiday',
        'weekday', 'workingday', 'weathersit'
    ]

    # Cek kolom wajib
    missing_cols = [col for col in categorical_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Kolom tidak lengkap: {missing_cols}")
        st.stop()

    df_processed = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Samakan kolom dengan training
    for col in training_columns:
        if col not in df_processed:
            df_processed[col] = 0

    # Urutkan kolom
    df_processed = df_processed[training_columns]

    # ========================
    # Prediksi
    # ========================
    try:
        predictions = model.predict(df_processed)
        df['prediksi'] = predictions

        st.write("Hasil Prediksi:", df)

        # Download
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download hasil",
            csv,
            "hasil_prediksi.csv",
            "text/csv"
        )

    except Exception as e:
        st.error(f"Terjadi error saat prediksi: {e}")
