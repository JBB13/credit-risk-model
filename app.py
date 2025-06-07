# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# Cargar modelo entrenado
# -----------------------------
model = joblib.load("modelo_credito.pkl")  # Asegúrate de haberlo guardado antes

# -----------------------------
# Columnas esperadas por el modelo
# -----------------------------
expected_features = [
    'RevolvingUtilizationOfUnsecuredLines',
    'age',
    'NumberOfTime30-59DaysPastDueNotWorse',
    'DebtRatio',
    'MonthlyIncome',
    'NumberOfOpenCreditLinesAndLoans',
    'NumberOfTimes90DaysLate',
    'NumberRealEstateLoansOrLines',
    'NumberOfTime60-89DaysPastDueNotWorse',
    'NumberOfDependents'
]

# -----------------------------
# Título e introducción
# -----------------------------
st.title("Calculadora de Riesgo de Crédito")

st.markdown("""
Esta app calcula la **Probabilidad de Default (PD)** y la **Pérdida Esperada (Expected Loss)** para clientes individuales.
""")

# -----------------------------
# Entrada de archivo
# -----------------------------
st.header("1. Datos del cliente")

uploaded_file = st.file_uploader("Sube un archivo CSV con las variables del cliente", type="csv")

# -----------------------------
# Entrada manual si no hay CSV
# -----------------------------
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Limpiar columnas extra si existen
    missing_cols = set(expected_features) - set(df.columns)
    if missing_cols:
        st.error(f"❌ El CSV está incompleto. Faltan columnas: {missing_cols}")
        st.stop()

    df = df[expected_features]
    st.success("Archivo CSV cargado correctamente")
else:
    st.info("No se subió archivo. Introduce los datos manualmente:")

    revolving_util = st.number_input("Utilización Revolvente de Líneas No Seguras", min_value=0.0, max_value=10.0, value=0.6)
    edad = st.number_input("Edad", min_value=18, max_value=100, value=45)
    atraso_30 = st.number_input("Veces con atraso 30-59 días", min_value=0, value=0)
    debt_ratio = st.number_input("Ratio de Deuda", min_value=0.0, max_value=100.0, value=0.3)
    ingresos = st.number_input("Ingreso Mensual", min_value=0, value=5000)
    cuentas_abiertas = st.number_input("Líneas de crédito abiertas", min_value=0, value=7)
    atraso_90 = st.number_input("Veces con atraso >90 días", min_value=0, value=0)
    prestamos_inmobiliarios = st.number_input("Préstamos de bienes raíces", min_value=0, value=1)
    atraso_60 = st.number_input("Veces con atraso 60-89 días", min_value=0, value=0)
    dependientes = st.number_input("Número de dependientes", min_value=0, value=2)


df = pd.DataFrame([{
    'RevolvingUtilizationOfUnsecuredLines': revolving_util,
    'age': edad,
    'NumberOfTime30-59DaysPastDueNotWorse': atraso_30,
    'DebtRatio': debt_ratio,
    'MonthlyIncome': ingresos,
    'NumberOfOpenCreditLinesAndLoans': cuentas_abiertas,
    'NumberOfTimes90DaysLate': atraso_90,
    'NumberRealEstateLoansOrLines': prestamos_inmobiliarios,
    'NumberOfTime60-89DaysPastDueNotWorse': atraso_60,
    'NumberOfDependents': dependientes
}])

# -----------------------------
# Entrada de LGD y EAD
# -----------------------------
st.header("2. Parámetros financieros")

lgd = st.slider("LGD (pérdida en caso de default) [%]", min_value=0, max_value=100, value=60)
ead = st.number_input("EAD (exposición al default) [$]", min_value=0.0, value=10000.0)

# -----------------------------
# Cálculo y Resultados
# -----------------------------
import joblib

# Cargar modelo y scaler
model = joblib.load("modelo_credito.pkl")
scaler = joblib.load("scaler_credito.pkl")

...

if st.button("Calcular PD y Expected Loss"):

    # Reordenar columnas si es necesario
    df = df[expected_features]

    # Escalar los datos de entrada
    df_scaled = scaler.transform(df)   # ✅ AQUÍ debes aplicar transform, no al modelo

    # Hacer la predicción con el modelo ya entrenado
    pd_values = model.predict_proba(df_scaled)[:, 1]
    df['PD'] = pd_values

    # Cálculo financiero
    df['LGD'] = lgd / 100
    df['EAD'] = ead
    df['Expected_Loss'] = df['PD'] * df['LGD'] * df['EAD']

    # Mostrar resultados
    st.subheader("3. Resultados del cliente")
    st.dataframe(df[['PD', 'LGD', 'EAD', 'Expected_Loss']])
    st.success(f"📊 Pérdida Esperada Total: ${df['Expected_Loss'].sum():,.2f}")
