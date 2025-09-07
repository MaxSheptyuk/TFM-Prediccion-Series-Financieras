import pandas as pd


# 1. Cargar el dataset
df = pd.read_csv("DATA/Dataset_All_Features.csv", sep=';', parse_dates=['Fecha'])


# Seleccionar las columnas de interés
cols_a_mostrar = [
    'Fecha',
    'Symbol',
    'Open',
    'Close',
    'High',
    'Low',
    'Volume',
    'EMA_10',
    'RSI_14',
    'CCI_14',
    'ROC_10',
    'TARGET_TREND_ANG_15_5'
]

# Mostrar las primeras 10 filas
print(df[cols_a_mostrar].head(10))