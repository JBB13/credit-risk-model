# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# Cargar modelo entrenado
# -----------------------------
model = joblib.load("modelo_credito.pkl")  # Asegúrate de guardar antes tu modelo

# -----------------------------
# Interfaz
# -----------------------------
st.title("Calculadora de Riesgo de Crédito")

st.markdown("""
Esta app calcula la **Probabilidad de Default (PD)** y la **Pérdida Esperada (Expected Loss)** para clientes individuales.
""")

# -----------------------------
# Subida o simulación de datos
# -----------------------------
st.header("1. Subir datos del cliente")

uploaded_file = st.file_uploader("Sube un archivo CSV con las variables del cliente", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.info("O usa los datos simulados")
    df = pd.DataFrame({
        'RevolvingUtilizationOfUnsecuredLines': [0.6],
        'age': [45],
        'NumberOfTime30-59DaysPastDueNotWorse': [0],
        'DebtRatio': [0.3],
        'MonthlyIncome': [5000],
        'NumberOfOpenCreditLinesAndLoans': [7],
        'NumberOfTimes90DaysLate': [0],
        'NumberRealEstateLoansOrLines': [1],
        'NumberOfTime60-89DaysPastDueNotWorse': [0],
        'NumberOfDependents': [2]
    })

st.write("### Datos del cliente")
st.dataframe(df)

# -----------------------------
# Entrada de LGD y EAD
# -----------------------------
st.header("2. Parámetros Financieros")

lgd = st.slider("LGD (Loss Given Default) [%]", min_value=0, max_value=100, value=60)
ead = st.number_input("EAD (Exposure at Default) [$]", min_value=0.0, value=10000.0)

# -----------------------------
# Predicción y cálculo
# -----------------------------
st.header("3. Resultados")

if st.button("Calcular PD y EL"):

    # Predecir PD
    pd_values = model.predict_proba(df)[:, 1]  # probabilidades de clase 1
    df['PD'] = pd_values
    df['LGD'] = lgd / 100
    df['EAD'] = ead
    df['Expected_Loss'] = df['PD'] * df['LGD'] * df['EAD']

    st.write("### Resultados por cliente")
    st.dataframe(df[['PD', 'LGD', 'EAD', 'Expected_Loss']])

    # Sumar EL total
    st.success(f"🧾 Pérdida Esperada Total: ${df['Expected_Loss'].sum():,.2f}")
