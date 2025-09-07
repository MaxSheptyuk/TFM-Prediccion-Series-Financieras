import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Carga dataset ---
df = pd.read_csv("DATA/Dataset_All_Features.csv", parse_dates=['Fecha'],   sep=';')

window = 10
horizon = 5

# --- Funciones de cálculo ---
def slope_window(arr):
    x = np.arange(len(arr))
    m = np.polyfit(x, arr, 1)[0]
    return m

def slope_window_normalized(arr):
    arr = np.asarray(arr)
    if arr.max() != arr.min():
        arr = (arr - arr.min()) / (arr.max() - arr.min())
    else:
        arr = np.zeros_like(arr)
    x = np.arange(len(arr))
    m = np.polyfit(x, arr, 1)[0]
    return m

# --- Sigmoid bipolar ---
def sigmoid_bipolar(x):
    # Clipping para evitar explosiones numéricas
    x = np.clip(x, -10, 10)
    return 0.5 + (((2 / (1 + np.exp(-x))) - 1) / 2)

# --- Bucle por símbolo ---
slopes = []
slopes_norm = []
slopes_clip = []
slopes_sigmoid = []

for symbol in df['Symbol'].unique():
    d = df[df['Symbol'] == symbol].reset_index(drop=True)
    closes = d['Close'].values
    for i in range(len(closes)):
        start = i - (window - horizon)
        end = i + horizon
        if start < 0 or end > len(closes):
            slopes.append(np.nan)
            slopes_norm.append(np.nan)
            slopes_clip.append(np.nan)
            slopes_sigmoid.append(np.nan)
        else:
            s_raw = slope_window(closes[start:end])
            s_norm = slope_window_normalized(closes[start:end])
            s_clip = np.clip(s_raw, -1, 1)  # clipping para slopes brutos [-1, 1]
            s_sigmoid = sigmoid_bipolar(s_clip)  # aplica sigmoid bipolar sobre la pendiente ya clippeada
            slopes.append(s_raw)
            slopes_norm.append(s_norm)
            slopes_clip.append(s_clip)
            slopes_sigmoid.append(s_sigmoid)

df['TARGET_RAW'] = slopes
df['TARGET_SLOPE_NORM'] = slopes_norm
df['TARGET_SLOPE_CLIP'] = slopes_clip
df['TARGET_SIGMOID_BIPOLAR'] = slopes_sigmoid

df2 = df.dropna(subset=['TARGET_RAW', 'TARGET_SLOPE_NORM', 'TARGET_SLOPE_CLIP', 'TARGET_SIGMOID_BIPOLAR'])

# Estadísticas para comparar
print("Pendiente RAW  : min {:.2f}, max {:.2f}, mean {:.2f}, std {:.2f}".format(
    df2['TARGET_RAW'].min(), df2['TARGET_RAW'].max(), df2['TARGET_RAW'].mean(), df2['TARGET_RAW'].std()))
print("Pendiente NORM : min {:.4f}, max {:.4f}, mean {:.4f}, std {:.4f}".format(
    df2['TARGET_SLOPE_NORM'].min(), df2['TARGET_SLOPE_NORM'].max(), df2['TARGET_SLOPE_NORM'].mean(), df2['TARGET_SLOPE_NORM'].std()))
print("Pendiente CLIP : min {:.2f}, max {:.2f}, mean {:.2f}, std {:.2f}".format(
    df2['TARGET_SLOPE_CLIP'].min(), df2['TARGET_SLOPE_CLIP'].max(), df2['TARGET_SLOPE_CLIP'].mean(), df2['TARGET_SLOPE_CLIP'].std()))
print("Sigmoid Bipolar: min {:.4f}, max {:.4f}, mean {:.4f}, std {:.4f}".format(
    df2['TARGET_SIGMOID_BIPOLAR'].min(), df2['TARGET_SIGMOID_BIPOLAR'].max(), df2['TARGET_SIGMOID_BIPOLAR'].mean(), df2['TARGET_SIGMOID_BIPOLAR'].std()))

# Visualización de las distribuciones
fig, axes = plt.subplots(2, 2, figsize=(15,10))

axes[0,0].hist(df2['TARGET_RAW'], bins=100, color='tab:blue', alpha=0.7)
axes[0,0].set_title('Histograma pendiente RAW')
axes[0,0].set_xlabel('Pendiente RAW')
axes[0,0].set_ylabel('Frecuencia')

axes[0,1].hist(df2['TARGET_SLOPE_NORM'], bins=100, color='tab:orange', alpha=0.7)
axes[0,1].set_title('Histograma pendiente normalizada [0,1]')
axes[0,1].set_xlabel('Pendiente Normalizada')
axes[0,1].set_ylabel('Frecuencia')

axes[1,0].hist(df2['TARGET_SLOPE_CLIP'], bins=100, color='tab:green', alpha=0.7)
axes[1,0].set_title('Histograma pendiente con clipping [-1,1]')
axes[1,0].set_xlabel('Pendiente Clipped')
axes[1,0].set_ylabel('Frecuencia')

axes[1,1].hist(df2['TARGET_SIGMOID_BIPOLAR'], bins=100, color='tab:red', alpha=0.7)
axes[1,1].set_title('Histograma pendiente + clipping + sigmoid_bipolar')
axes[1,1].set_xlabel('Sigmoid Bipolar')
axes[1,1].set_ylabel('Frecuencia')

plt.tight_layout()
plt.show()
