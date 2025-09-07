import numpy as np
import pandas as pd
from scipy.stats import kurtosis


# --- Carga dataset ---
df = pd.read_csv("DATA/Dataset_All_Features.csv", parse_dates=['Fecha'],   sep=';')

# Define features y target
features = [
    'RSI_14',         # momentum
    'MACD_12_26_9',   # oscilador
    'ATR_14',         # volatilidad
    'OBV',            # volumen
    'CCI_14',         # momentum alternativo
    'CONNORS_RSI_3_2_25',  # mixto
    'BB_WIDTH_20_2',       # ancho Bollinger (volatilidad)
    'CHO_3_10',            # Chaikin Oscillator
]

df['symbol_change'] = df['Symbol'] != df['Symbol'].shift(1)
df['OBV_jump'] = df['OBV'].diff().abs()
print(df.loc[df['symbol_change'], ['Symbol', 'Fecha', 'OBV', 'OBV_jump']].head(20))