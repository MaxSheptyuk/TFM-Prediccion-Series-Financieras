import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ======================================================
#      VISUALIZACIÓN DE PREDICCIÓN DE TENDENCIA
# ======================================================
#
# Este script dibuja el precio de cierre real de NVDA y
# las proyecciones de tendencia generadas por el modelo
# (líneas naranjas rectas) desde distintos puntos t
# (círculos), cada cierto número de días.
#
# - Cada línea naranja es una proyección recta a partir de t,
#   usando el ángulo de tendencia predicho en ese punto
#   (no utiliza precios futuros).
# - La duración de la línea es configurable.
# - Las proyecciones se espacian cada "step" días bursátiles.
#
# ------------------------------------------------------

# --- 1. CARGA DE DATOS --------------------------------

# Cargar precios históricos
df_prices = pd.read_csv("DATA/AllStocksHistoricalData.csv", sep=';', parse_dates=['Fecha'])
# Filtrar solo el stock NVDA y ordenar por fecha
df_prices = df_prices[df_prices['Symbol'] == 'NVDA'].sort_values('Fecha').reset_index(drop=True)

# Cargar las predicciones de tendencia (ángulo normalizado)
df_pred = pd.read_csv("Resultados/Predicciones_Test_XGBRegressor_NVDA.csv", parse_dates=['Fecha'])

# --- 2. SELECCIÓN DE PERIODO A VISUALIZAR -------------

fecha_inicio = pd.to_datetime("2023-06-01")
fecha_fin    = pd.to_datetime("2024-02-01")

# Filtrar precios y predicciones en el rango elegido
mask_prices = (df_prices['Fecha'] >= fecha_inicio) & (df_prices['Fecha'] <= fecha_fin)
mask_pred = (df_pred['Fecha'] >= fecha_inicio) & (df_pred['Fecha'] <= fecha_fin)

df_prices_filtrado = df_prices[mask_prices].copy()
df_pred_filtrado = df_pred[mask_pred].copy()

# Unir ambos dataframes por fecha (inner join)
df_merge = pd.merge(df_prices_filtrado, df_pred_filtrado, on="Fecha", how="inner")

# --- 3. PARÁMETROS DE VISUALIZACIÓN -------------------

# Contexto (solo como referencia, no se usa aquí)
window = 15              # Tamaño de la ventana para features (solo contexto)
horizonte = 5            # Horizonte real de predicción (ej. target t+5)
pred_line_length = 10     # Duración (en días bursátiles) de cada línea de predicción
step = 15                # Espaciado entre cada predicción mostrada

# Paleta de colores personalizada
color_precio = "#24397A"    # Azul muy oscuro para precio real
color_pred = "#ff9900"      # Naranja fuerte para predicción modelo

# --- 4. GRÁFICO ----------------------------------------

sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(14, 6))

# 4.1. Dibujar la línea del precio de cierre (azul oscuro)
ax.plot(
    df_merge['Fecha'], 
    df_merge['Close'], 
    color=color_precio, 
    linewidth=2, 
    label="Precio de cierre NVDA", 
    zorder=3
)

# 4.2. Dibujar líneas de predicción rectas, cada "step" días
# El bucle recorre solo los índices donde hay suficientes días futuros para proyectar la línea
ultimo_idx = len(df_merge) - pred_line_length

for i in range(0, ultimo_idx, step):
    # Índice del punto de partida de la predicción
    idx_inicio = i

    # --- PUNTO DE INICIO DE LA PROYECCIÓN ---
    fecha_inicio_pred = df_merge['Fecha'].iloc[idx_inicio]
    precio_inicio_pred = df_merge['Close'].iloc[idx_inicio]
    angle_norm = df_merge['Predicted'].iloc[idx_inicio]

    # Convertir ángulo normalizado [0,1] a grados [-75º, 75º]
    angle_degrees = (angle_norm - 0.5) * 2 * 75
    # Obtener pendiente (m) a partir del ángulo
    slope = np.tan(np.deg2rad(angle_degrees))

    # --- PUNTO FINAL DE LA PROYECCIÓN ---
    idx_final = idx_inicio + pred_line_length
    if idx_final >= len(df_merge):
        break  # No hay suficientes días, salimos del bucle

    fecha_final_pred = df_merge['Fecha'].iloc[idx_final]
    precio_final_pred = precio_inicio_pred + slope * pred_line_length

    # --- DIBUJAR LA LÍNEA DE PREDICCIÓN ---
    ax.plot(
        [fecha_inicio_pred, fecha_final_pred],     # Eje X: fechas inicial y final
        [precio_inicio_pred, precio_final_pred],   # Eje Y: precios inicial y final proyectados
        color=color_pred,
        lw=3.1,
        linestyle="--",
        alpha=0.93,
        label="Predicción modelo" if i == 0 else "",  # Solo la primera vez para leyenda
        zorder=2
    )

    # --- MARCAR EL PUNTO DE INICIO (círculo naranja) ---
    ax.scatter(
        fecha_inicio_pred,
        precio_inicio_pred,
        color=color_pred,
        s=80,
        zorder=4,
        edgecolor="k",
        linewidth=1.1
    )

# --- 5. FORMATO FINAL DEL GRÁFICO ----------------------

ax.legend(fontsize=13, frameon=True, loc="upper left")
ax.set_xlabel("Fecha", fontsize=13)
ax.set_ylabel("Precio cierre", fontsize=13)
ax.set_title(
    "NVDA: Precio de cierre y predicción modelo (líneas rectas)\n(ventana=15d, horizonte=5d)",
    fontsize=17, pad=13
)
ax.grid(axis='y', linestyle=':', alpha=0.42)
ax.set_axisbelow(True)
plt.tight_layout()
plt.show()
