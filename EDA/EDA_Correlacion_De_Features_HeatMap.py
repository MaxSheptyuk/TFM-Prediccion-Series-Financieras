import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Carga dataset ---
df = pd.read_csv("DATA/Dataset_All_Features.csv", parse_dates=['Fecha'],   sep=';')


cols = [
    'EMA_10', 'EMA_14', 'ADX_14', 'RSI_14', 'CCI_14', 'MACD_12_26_9', 'MACD_HISTOGRAM_12_26_9',
    'STOCH_K_14_3_3', 'WILLIAMS_R_14', 'ATR_14', 'BB_UPPER_20_2', 'BB_WIDTH_20_2', 'OBV',
    'CONNORS_RSI_3_2_25', 'UO_7_14_28', 'CHO_3_10', 'ULCER_INDEX_14' 
]
cols = [col for col in cols if col in df.columns]

corr = df[cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))

plt.figure(figsize=(18, 13))  # Más grande
ax = sns.heatmap(
    corr.round(2),           # Redondear a 2 decimales
    mask=mask,
    annot=True,
    fmt='.2f',               # Solo 2 decimales
    annot_kws={"size": 9},  # Letra un poco más pequeña
    cmap='coolwarm',
    vmin=-1, vmax=1,
    linewidths=0.5,
    square=True,
    cbar_kws={"shrink": .8}
)
plt.title('Matríz de correlación entre indicadores técnicos')
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()