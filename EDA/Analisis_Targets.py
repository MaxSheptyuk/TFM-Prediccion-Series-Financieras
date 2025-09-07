import os
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt

# ====== CONFIG ======
CSV_PATH      = "DATA/Dataset_All_Features_Transformado.csv"  
DATE_COL      = "Fecha"
SYMBOL_COL    = "Symbol"
TARGET_COL    = "TARGET_TREND_ANG_15_5"
START_DATE    = "2010-01-01"
END_DATE_EXCL = "2020-01-01"
OUT_DIR       = "Resultados/Target_Histograms_2010_2019"
BINS          = 40
HIST_RANGE    = (0.0, 1.0)  # si tu target está en [0,1]

os.makedirs(OUT_DIR, exist_ok=True)

# ====== CARGA Y FILTRO ======
df = pd.read_csv(CSV_PATH, sep=';', parse_dates=[DATE_COL])
df = df[(df[DATE_COL] >= START_DATE) & (df[DATE_COL] < END_DATE_EXCL)].copy()

# Si ya tienes la lista exacta de 25 símbolos, colócala aquí:
# symbols = ["AAPL","NVDA",...]
# Si no, tomamos los 25 más frecuentes:
symbols = (df[SYMBOL_COL].value_counts()
           .sort_values(ascending=False)
           .head(25).index.tolist())

# ====== FUNCIONES AUX ======
def eda_series(y: pd.Series) -> dict:
    y = y.dropna().astype(float)
    if len(y) == 0:
        return None
    return {
        "n":            int(len(y)),
        "mean":         float(y.mean()),
        "std":          float(y.std(ddof=1)),
        "p10":          float(y.quantile(0.10)),
        "p25":          float(y.quantile(0.25)),
        "p50":          float(y.quantile(0.50)),
        "p75":          float(y.quantile(0.75)),
        "p90":          float(y.quantile(0.90)),
        "skew":         float(skew(y, bias=False)),
        "kurt_excess":  float(kurtosis(y, fisher=True, bias=False)),
        "frac_045_055": float(((y>=0.45) & (y<=0.55)).mean()),
        "frac_lt_025":  float((y<0.25).mean()),
        "frac_gt_075":  float((y>0.75).mean()),
        "delta_mean_0_5": float(y.mean() - 0.5)
    }

# ====== LOOP POR SÍMBOLO: HIST + STATS ======
rows = []
for sym in symbols:
    s = df.loc[df[SYMBOL_COL]==sym, TARGET_COL].dropna().astype(float)
    if s.empty: 
        continue

    # Histograma (una figura por símbolo)
    plt.figure()
    plt.hist(s.values, bins=BINS, range=HIST_RANGE)
    plt.title(f"{sym} — {TARGET_COL} (2010–2019)")
    plt.xlabel(TARGET_COL); plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"hist_{sym}_{TARGET_COL}_2010_2019.png"))
    plt.close()

    stats = eda_series(s)
    stats.update({"Symbol": sym})
    rows.append(stats)

# ====== RESUMEN GLOBAL ======
summary = pd.DataFrame(rows).set_index("Symbol").sort_values("mean")
summary_path = os.path.join(OUT_DIR, "target_hist_summary_2010_2019.csv")
summary.to_csv(summary_path, float_format="%.6f")

# También, distribución "pooled" de todos los símbolos:
all_y = df.loc[df[SYMBOL_COL].isin(symbols), TARGET_COL].dropna().astype(float)
plt.figure()
plt.hist(all_y.values, bins=BINS, range=HIST_RANGE)
plt.title(f"ALL (25) — {TARGET_COL} (2010–2019)")
plt.xlabel(TARGET_COL); plt.ylabel("Frecuencia")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f"hist_ALL_{TARGET_COL}_2010_2019.png"))
plt.close()

print(f"Guardado resumen en: {summary_path}")
print(f"PNG de histogramas por símbolo en: {OUT_DIR}")
