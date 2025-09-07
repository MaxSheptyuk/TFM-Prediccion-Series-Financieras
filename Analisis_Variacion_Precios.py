# -*- coding: utf-8 -*-
"""
Magnitud de variaciones en horizonte de 5 días (2020–2024)
----------------------------------------------------------
Calcula estadísticos descriptivos de los rendimientos acumulados absolutos
a 5 días para cada activo en SYMBOLS_TO_TEST.
"""

import pandas as pd
import numpy as np

# Ruta al dataset con precios
DATASET_PATH = "DATA/Dataset_All_Features_Transformado.csv"

# Símbolos a analizar (los 4 principales en tu texto)
SYMBOLS_TO_TEST = ["NVDA", "MSFT", "MU", "AAPL"]

# Horizonte en días
H = 5

def calcular_variaciones(df: pd.DataFrame, symbol: str, h: int = H) -> pd.DataFrame:
    """Devuelve DF con magnitud del rendimiento acumulado a h días."""
    d = df[df["Symbol"] == symbol].copy()
    d = d.sort_values("Fecha").reset_index(drop=True)

    # Rendimiento acumulado a h días
    ret_col = f"ret_{h}d"
    d[ret_col] = (d["Close"].shift(-h) - d["Close"]) / d["Close"]
    d = d.dropna(subset=[ret_col])
    d[f"abs_{ret_col}"] = d[ret_col].abs()
    return d

def resumen_estadistico(series: pd.Series) -> dict:
    """Devuelve dict con estadísticas clave de una serie."""
    return {
        "media":   series.mean(),
        "mediana": series.median(),
        "std":     series.std(),
        "p05":     series.quantile(0.05),
        "p25":     series.quantile(0.25),
        "p75":     series.quantile(0.75),
        "p95":     series.quantile(0.95),
    }

def main():
    # Leer dataset y filtrar 2020–2024
    df = pd.read_csv(DATASET_PATH, sep=";", parse_dates=["Fecha"])
    df = df[df["Fecha"].dt.year.between(2020, 2024)]

    resultados_abs = []
    for sym in SYMBOLS_TO_TEST:
        d = calcular_variaciones(df, sym, H)
        ret_col = f"abs_ret_{H}d"
        stats_abs = resumen_estadistico(d[ret_col])
        stats_abs["Symbol"] = sym
        resultados_abs.append(stats_abs)

    # Tabla de magnitudes
    tabla_abs = pd.DataFrame(resultados_abs)[
        ["Symbol", "media", "mediana", "std", "p05", "p25", "p75", "p95"]
    ]

    print("\n=== Magnitud de variaciones |R_{t,5}| (2020–2024) ===")
    print(tabla_abs.to_string(index=False, float_format=lambda x: f"{x:.3%}"))

if __name__ == "__main__":
    main()
