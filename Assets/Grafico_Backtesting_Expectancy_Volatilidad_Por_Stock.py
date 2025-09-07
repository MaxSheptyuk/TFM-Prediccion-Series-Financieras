# -*- coding: utf-8 -*-
"""
Scatter Expectancy vs Volatilidad por símbolo
- X = desviación estándar de retornos por trade (%)
- Y = expectancy medio (%)
Cada punto = un ticker.
"""

import pandas as pd
import matplotlib.pyplot as plt

# === Cargar log de trading ===
log_path = "Resultados/Trading_Log_AllStocks_TARGET_TREND_ANG_5_15.csv"  # <-- ajusta la ruta
df = pd.read_csv(log_path, sep=";", parse_dates=["Fecha"])

# Solo operaciones de cierre
df_sell = df[df["Accion"].str.startswith("SELL")].copy()

# Capital anterior = capital actual - profit
df_sell["Capital_Anterior"] = df_sell["Capital_Actual"] - df_sell["Profit"]

# Retorno % por trade
df_sell["Retorno_%"] = df_sell["Profit"] / df_sell["Capital_Anterior"] * 100

# Agrupar por símbolo
summary = (
    df_sell.groupby("Symbol")
           .agg(Expectancy_pct=("Retorno_%", "mean"),
                Volatility_pct=("Retorno_%", "std"),
                N_trades=("Retorno_%", "count"))
           .reset_index()
)

# === Scatter plot ===
fig, ax = plt.subplots(figsize=(9,6))
ax.scatter(summary["Volatility_pct"], summary["Expectancy_pct"], s=40+summary["N_trades"])

# Etiquetas de cada ticker
for _, row in summary.iterrows():
    ax.text(row["Volatility_pct"]+0.05, row["Expectancy_pct"],
            row["Symbol"], fontsize=9, ha="left", va="center")

# Línea de referencia
ax.axhline(0, color="gray", linestyle="--", linewidth=1)

ax.set_xlabel("Volatilidad de retornos por trade (%)")
ax.set_ylabel("Expectancy medio por trade (%)")
ax.set_title("Expectancy vs Volatilidad por símbolo — Eficiencia de activos")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
