# BackTesting/BacktesterCalidadML.py
# ---------------------------------------------------------------
# Evalúa la "calidad de señal" sin TP/SL ni capital:
#   - Añadido: métricas acumuladas (suma de oportunidades, hit ratio).
# ---------------------------------------------------------------

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass

@dataclass
class CalidadConfig:
    horizon: int = 5
    threshold_buy: float = 0.51
    signal_colname: str = "y_hat"
    price_colname: str = "Close"
    date_colname: str = "Fecha"
    symbol_colname: str = "Symbol"

class BacktesterCalidadML:
    def __init__(self, cfg: CalidadConfig):
        self.cfg = cfg

    @staticmethod
    def _spearman_safe(x, y):
        x = pd.Series(x).astype(float)
        y = pd.Series(y).astype(float)
        if len(x) < 3 or x.nunique() < 2 or y.nunique() < 2:
            return np.nan
        return x.rank(method="average").corr(y.rank(method="average"))

    def _path_metrics_up(self, closes: np.ndarray) -> dict:
        p0 = closes[0]
        fw = (closes[1:] - p0) / p0  # retornos relativos
        pos = fw[fw > 0]
        avg_pos = float(pos.mean()) if pos.size > 0 else 0.0
        return {
            "final_ret": float(fw[-1]),
            "max_path_ret": float(fw.max()),
            "avg_path_ret": float(fw.mean()),
            "sum_pos_path_ret": float(pos.sum()) if pos.size > 0 else 0.0,
            "avg_pos_path_ret": avg_pos,
            "n_pos_days": int(pos.size),
        }

    def run(self, df_test: pd.DataFrame, y_hat: np.ndarray, symbol: str) -> dict:
        cfg = self.cfg
        df = df_test.copy().reset_index(drop=True)
        df[cfg.signal_colname] = y_hat.astype(float)

        entries = df[df[cfg.signal_colname] > cfg.threshold_buy].copy()
        if entries.empty:
            return {"trade_log": pd.DataFrame(), "metrics": {}}

        rows = []
        close = df[cfg.price_colname].to_numpy()
        dates = df[cfg.date_colname].to_numpy()
        H = cfg.horizon

        for idx in entries.index.to_numpy():
            if idx + H >= len(df):
                continue
            path_closes = close[idx: idx + H + 1]
            stats = self._path_metrics_up(path_closes)
            rows.append({
                "Symbol": symbol,
                "Fecha": dates[idx],
                "Entrada": float(close[idx]),
                "H": H,
                "Signal": float(df.loc[idx, cfg.signal_colname]),
                **stats
            })

        trade_log = pd.DataFrame(rows)
        if trade_log.empty:
            return {"trade_log": trade_log, "metrics": {}}

        # ---- Métricas de correlación ----
        ic_final = trade_log["Signal"].corr(trade_log["final_ret"])
        ric_final = self._spearman_safe(trade_log["Signal"], trade_log["final_ret"])

        # ---- Métricas promedio (como antes) ----
        mean_final = float(trade_log["final_ret"].mean())
        median_final = float(trade_log["final_ret"].median())
        mean_avgpos = float(trade_log["avg_pos_path_ret"].mean())

        # ---- NUEVAS métricas acumuladas ----
        total_sum_pos_path = float(trade_log["sum_pos_path_ret"].sum())
        total_pos_days = int(trade_log["n_pos_days"].sum())
        hit_ratio = float((trade_log["n_pos_days"] > 0).mean())  # % con algún día positivo

        metrics = {
            "n_trades": len(trade_log),

            # Medias
            "mean_final_ret": mean_final,
            "median_final_ret": median_final,
            "mean_avg_pos_path_ret": mean_avgpos,

            # Acumulados
            "total_sum_pos_path_ret": total_sum_pos_path,
            "total_pos_days": total_pos_days,
            "hit_ratio": hit_ratio,

            # Correlaciones
            "IC_final": float(ic_final) if pd.notna(ic_final) else np.nan,
            "Rank_IC_final": float(ric_final) if pd.notna(ric_final) else np.nan,
        }

        return {"trade_log": trade_log, "metrics": metrics}
