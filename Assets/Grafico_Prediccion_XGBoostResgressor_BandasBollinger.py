import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import MaxNLocator

# Carga y filtra
df = pd.read_csv("Resultados/Predicciones_Test_XGBRegressor_NVDA.csv", parse_dates=["Fecha"])
df = df.sort_values("Fecha").reset_index(drop=True)

fecha_inicio = pd.to_datetime("2023-03-01")
fecha_fin    = pd.to_datetime("2023-05-31")
df_filtrado = df[(df['Fecha'] >= fecha_inicio) & (df['Fecha'] <= fecha_fin)].copy()

sns.set_theme(style="whitegrid", palette="muted")
colores = sns.color_palette("muted")

window = 20
# Bandas sobre valor real (target)
df_filtrado['mean_real'] = df_filtrado['Target'].rolling(window, center=True, min_periods=1).mean()
df_filtrado['std_real'] = df_filtrado['Target'].rolling(window, center=True, min_periods=1).std()
df_filtrado['band_upper'] = df_filtrado['mean_real'] + 2 * df_filtrado['std_real']
df_filtrado['band_lower'] = df_filtrado['mean_real'] - 2 * df_filtrado['std_real']

y_real = df_filtrado['Target'].values
y_pred = df_filtrado['Predicted'].values
x = df_filtrado['Fecha'].values
errors = y_pred - y_real

plt.figure(figsize=(13, 6))

# Banda ±2σ sobre valor real (target)
plt.fill_between(df_filtrado['Fecha'], df_filtrado['band_lower'], df_filtrado['band_upper'],
                 color=colores[0], alpha=0.18, label='Banda ±2σ (target)', zorder=1)

# Línea valor real (gris oscuro, sólida)
plt.plot(df_filtrado['Fecha'], y_real, color="#232323", linewidth=2.5, label='Ángulo trend, horizonte=5d (real)', zorder=3)

# Línea predicha (color seaborn, discontinua)
plt.plot(df_filtrado['Fecha'], y_pred, linestyle='--', color=colores[1], linewidth=2, alpha=0.93, label='Predicción modelo', zorder=4)

# Barras verticales coloreadas por error (con suavidad)
for i in range(len(x)):
    color = colores[4] if errors[i] < 0 else colores[3]
    plt.vlines(x[i], y_real[i], y_pred[i], color=color, linewidth=2, alpha=0.6, zorder=2)

plt.xlabel("Fecha", fontsize=13)
plt.ylabel("Ángulo trend, horizonte=5d", fontsize=13)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=False))
plt.grid(axis='y', linestyle=':', alpha=0.44)
plt.grid(axis='x', which='major', linestyle='', alpha=0.0)

titulo_fechas = f"{fecha_inicio.strftime('%d/%m/%Y')} a {fecha_fin.strftime('%d/%m/%Y')}"
plt.title(f"NVDA · Ángulo trend, horizonte=5d: Real vs Predicho\n({titulo_fechas})", fontsize=16, pad=14)
plt.legend(fontsize=12, frameon=True, fancybox=True, shadow=False)
plt.tight_layout()
plt.subplots_adjust(top=0.88)

plt.show()
