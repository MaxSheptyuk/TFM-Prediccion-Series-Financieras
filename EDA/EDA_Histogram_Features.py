import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# --- Carga dataset ---
df = pd.read_csv("DATA/Dataset_All_Features.csv", parse_dates=['Fecha'], sep=';')

# Elige el símbolo para EDA (por ejemplo, 'AAPL')
symbol = 'AAPL'
df_stock = df[df['Symbol'] == symbol].copy()

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
target = 'TARGET_TREND_ANG_15_5'

# Filtra solo las columnas presentes
cols = [col for col in features if col in df_stock.columns]
if target in df_stock.columns:
    cols.append(target)

# Estandarizar solo los features (por stock)
scaler = StandardScaler()
df_std = pd.DataFrame(scaler.fit_transform(df_stock[features]), columns=features, index=df_stock.index)
df_std[target] = df_stock[target]  # Añade la variable objetivo original


# Plot de histogramas
df_std.hist(column=df_std.columns, bins=50, figsize=(14, 10), layout=(3, 3), sharex=False)
plt.suptitle(f'Histogramas de 8 indicadores técnicos (transformados) y target para {symbol}')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
