import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# Descargar datos de AAPL para los últimos 4 meses
aapl = yf.download('AAPL', period="4mo")
aapl = aapl.reset_index()

window = 15
num_segments = 3
step = (len(aapl) - window) // num_segments
segment_starts = [window - 1 + i*step for i in range(num_segments)]
segment_ends = [start + window - 1 for start in segment_starts]
horizon_offsets = [int(0.75*(end-start)) for start, end in zip(segment_starts, segment_ends)]
segment_horizons = [start + offset for start, offset in zip(segment_starts, horizon_offsets)]

vline_colors = ['#47a3e7', '#6b8366', '#ec407a']
vline_styles = ['dotted', 'dashed', (0, (3, 1, 1, 1))]
fill_colors = ['#c8e6ff', '#d4ffd4', '#fce1e6']

slopes = np.full(len(aapl), np.nan)
for i in range(window-1, len(aapl)):
    y = aapl['Close'].iloc[i-window+1:i+1].values
    x = np.arange(window)
    m, b = np.polyfit(x, y, 1)
    slopes[i] = m

fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(15, 8),
    sharex=True,
    gridspec_kw={'height_ratios': [2, 1]}
)

fontsize_titulo = 17
fontsize_ejes = 13
fontsize_leyenda = 12
fontsize_texto = 12
fontsize_horizonte = 11

# --- Gráfico superior: precios y tramos de regresión lineal ---
ax1.plot(aapl['Date'], aapl['Close'], color='blue', label='Cierre AAPL')
for idx, (start, end, horizon) in enumerate(zip(segment_starts, segment_ends, segment_horizons)):
    ax1.axvspan(
        aapl['Date'].iloc[horizon], aapl['Date'].iloc[end],
        color=fill_colors[idx], alpha=0.26, zorder=0
    )
    y = aapl['Close'].iloc[start:end+1].values
    if len(y) < window:
        continue
    x = np.arange(len(y))
    m, b = np.polyfit(x, y, 1)
    ax1.plot(
        [aapl['Date'].iloc[start], aapl['Date'].iloc[end]],
        [m*0 + b, m*(len(y)-1) + b],
        color='red', linewidth=2, label='Pendiente 15 días' if idx == 0 else ""
    )
    fecha_texto = aapl['Date'].iloc[start+1]
    y_max = np.max(aapl['Close'].iloc[start:end+1])
    ax1.text(
        fecha_texto,
        y_max + 5,
        'Variable objetivo',
        fontsize=fontsize_texto,
        color='darkred',
        ha='left', va='bottom', fontweight='normal', alpha=0.95
    )
    ax1.axvline(aapl['Date'].iloc[start], color=vline_colors[idx], linestyle=vline_styles[idx], alpha=0.8, linewidth=1)
    ax1.axvline(aapl['Date'].iloc[horizon], color=vline_colors[idx], linestyle=vline_styles[idx], alpha=0.8, linewidth=1)
    ax1.axvline(aapl['Date'].iloc[end], color=vline_colors[idx], linestyle=vline_styles[idx], alpha=0.8, linewidth=1)
    # --- AQUÍ SE BAJA EL TEXTO ROTADO ---
    x_middle = int((horizon + end)//2)
    x_text = aapl['Date'].iloc[x_middle]
    ymin, ymax = ax1.get_ylim()
    y_text = ymin + 0.76*(ymax-ymin)    # <--- BAJADO respecto a antes
    ax1.text(
        x_text, y_text, "Horizonte de predicción",
        fontsize=fontsize_horizonte, color=vline_colors[idx], ha='center', va='center',
        rotation=90, alpha=1, fontweight='normal'
    )

ax1.set_ylabel('Precio cierre ($)', fontsize=fontsize_ejes)
ax1.set_title('Precio de AAPL con tramos de regresión lineal (ventana 15 días, líneas de inicio, horizonte y fin)',
              fontsize=fontsize_titulo)
ax1.legend(loc='upper left', fontsize=fontsize_leyenda)

# --- Gráfico inferior: slope y sombreado SOLO en el horizonte ---
ax2.plot(aapl['Date'], slopes, color='black')
for idx, (start, end, horizon) in enumerate(zip(segment_starts, segment_ends, segment_horizons)):
    ax2.axvline(aapl['Date'].iloc[start], color=vline_colors[idx], linestyle=vline_styles[idx], alpha=0.7, linewidth=1)
    ax2.axvline(aapl['Date'].iloc[horizon], color=vline_colors[idx], linestyle=vline_styles[idx], alpha=0.7, linewidth=1)
    ax2.axvline(aapl['Date'].iloc[end], color=vline_colors[idx], linestyle=vline_styles[idx], alpha=0.7, linewidth=1)
    ax2.fill_between(
        aapl['Date'].iloc[horizon:end+1],
        slopes[horizon:end+1],
        0,
        color=fill_colors[idx],
        alpha=0.35
    )
ax2.set_ylabel('Slope (pendiente)', fontsize=fontsize_ejes)
ax2.set_xlabel('Fecha', fontsize=fontsize_ejes)
ax2.set_title('Evolución de la pendiente de regresión lineal (ventana móvil 15 días)',
              fontsize=fontsize_titulo-1)
ax2.axhline(0, color='gray', linewidth=0.7, linestyle=':')

plt.tight_layout()
plt.subplots_adjust(top=0.88, bottom=0.11)
plt.show()
