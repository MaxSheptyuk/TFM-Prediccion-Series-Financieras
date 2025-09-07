import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.interpolate import griddata

# --------- CONFIGURAMOS AQUÍ ----------
SYMBOLS = ['AAPL', 'MSFT', 'NVDA', 'CAT', 'NKE', 'REGN', 'BKNG']  
METRIC = "r2_out_of_sample"  # O 'mae_out_of_sample', 'r2_out_of_sample', 'mse_out_of_sample'
CSV_PATH_TEMPLATE = "Resultados/Benchmark_GA_Automatico_{}.csv"
# -----------------------------------

METRIC_INFO = {
    "rmse_out_of_sample": {
        "label": "Nivel de RMSE out-of-sample (más oscuro = mejor)",
        "cmap": "Purples_r",
        "title": "Mapa de niveles promedio de RMSE según nº de features y tamaño de dataset\n(XGBoost + GA, media de activos)"
    },
    "mae_out_of_sample": {
        "label": "Nivel de MAE out-of-sample (más oscuro = mejor)",
        "cmap": "PuRd_r",
        "title": "Mapa de niveles promedio de MAE según nº de features y tamaño de dataset\n(XGBoost + GA, media de activos)"
    },
    "mse_out_of_sample": {
        "label": "Nivel de MSE out-of-sample (más oscuro = mejor)",
        "cmap": "PuBuGn_r",
        "title": "Mapa de niveles promedio de MSE según nº de features y tamaño de dataset\n(XGBoost + GA, media de activos)"
    },
    "r2_out_of_sample": {
        "label": "Nivel de R² out-of-sample (más oscuro = mejor)",
        "cmap": "Purples",
        "title": "Mapa de niveles promedio de R² según nº de features y tamaño de dataset\n(XGBoost + GA, media de activos)"
    },
}

# --- Cargar y concatenar todos los CSVs ---
dfs = []
for sym in SYMBOLS:
    df = pd.read_csv(CSV_PATH_TEMPLATE.format(sym))
    df['symbol'] = sym  # Por si quieres filtrar luego
    dfs.append(df)
df_all = pd.concat(dfs, ignore_index=True)

# --- Calcular la media de la métrica por cada punto del grid ---
grouped = df_all.groupby(['n_features', 'dataset_size'])[METRIC].mean().reset_index()

X = grouped["n_features"].values
Y = grouped["dataset_size"].values
Z = grouped[METRIC].values

xi = np.linspace(X.min(), X.max(), 100)
yi = np.linspace(Y.min(), Y.max(), 100)
xi, yi = np.meshgrid(xi, yi)
zi = griddata((X, Y), Z, (xi, yi), method='cubic')

plt.figure(figsize=(10, 8))
contourf = plt.contourf(
    xi, yi, zi,
    levels=14,
    cmap=METRIC_INFO[METRIC]["cmap"]
)
contour = plt.contour(xi, yi, zi, levels=14, colors='black', linewidths=0.5)
plt.scatter(X, Y, color='mediumvioletred', s=50, label='Experimentos (media)', zorder=10, alpha=0.7)

plt.colorbar(contourf, label=METRIC_INFO[METRIC]["label"])
plt.xlabel("Número de features seleccionadas")
plt.ylabel("Tamaño del dataset (train)")
plt.title(METRIC_INFO[METRIC]["title"])
plt.legend()
plt.tight_layout()

# Formatear eje Y con separador de miles
ax = plt.gca()
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}".replace(",", ".")))

plt.show()
