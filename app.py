import streamlit as st
import pandas as pd
import joblib
import os


st.title("Prediksi Penyewaan Sepeda")


# =====================
# LOAD MODEL
# =====================

model = joblib.load(
    "model_xgb.pkl"
)

columns = joblib.load(
    "columns.pkl"
)


# =====================
# INPUT FILE
# =====================

file = st.file_uploader(
    "Upload file CSV",
    type="csv"
)


if file:

    df = pd.read_csv(file)

    st.subheader("Data Input")
    st.write(df.head())


    # =====================
    # PREPROCESSING
    # =====================

    categorical_cols = [
        'season',
        'yr',
        'mnth',
        'holiday',
        'weekday',
        'workingday',
        'weathersit'
    ]


    df_process = pd.get_dummies(
        df,
        columns=categorical_cols,
        drop_first=True
    )


    # Tambah kolom yang hilang

    for col in columns:
        if col not in df_process.columns:
            df_process[col] = 0


    # Urutkan kolom

    df_process = df_process[columns]


    # =====================
    # PREDIKSI
    # =====================

    hasil = model.predict(
        df_process
    )


    df["prediksi"] = hasil


    st.subheader("Hasil Prediksi")

    st.write(df)


    # Download hasil

    csv = df.to_csv(
        index=False
    ).encode("utf-8")


    st.download_button(
        "Download Hasil",
        csv,
        "hasil_prediksi.csv",
        "text/csv"
    )
