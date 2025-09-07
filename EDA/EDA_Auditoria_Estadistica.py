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

def feature_audit(df, features):
    summary = []
    for col in features:
        x = df[col].dropna()
        zeros = (x == 0).sum()
        zero_ratio = zeros / len(x)
        skew = x.skew()
        kurt = kurtosis(x, fisher=True)
        n_unique = x.nunique()
        top_val = x.value_counts().idxmax()
        top_val_ratio = (x == top_val).sum() / len(x)
        min_val, max_val = x.min(), x.max()
        summary.append({
            "feature": col,
            "zero_ratio": zero_ratio,
            "top_val": top_val,
            "top_val_ratio": top_val_ratio,
            "n_unique": n_unique,
            "skew": skew,
            "kurtosis": kurt,
            "min": min_val,
            "max": max_val
        })
    return pd.DataFrame(summary).sort_values("zero_ratio", ascending=False)

# Uso:
summary_df = feature_audit(df, features)
pd.set_option('display.max_rows', 50)
print(summary_df)
