# Rendimiento_Anual_Cartera.py (versión continua y robusta)
import pandas as pd
import numpy as np
import locale
from pathlib import Path

# ================== Localización ES ==================
try:
    locale.setlocale(locale.LC_ALL, 'es_ES.UTF-8')
except locale.Error:
    try:
        locale.setlocale(locale.LC_ALL, 'Spanish_Spain')
    except locale.Error:
        pass

def formato_espanol_numero(num, decimales=2, porcentaje=False):
    if isinstance(num, (int, float, np.floating)) and np.isfinite(num):
        s = f"{num:,.{decimales}f}"
        s = s.replace(",", "X").replace(".", ",").replace("X", ".")
        return f"{s}%" if porcentaje else s
    return "-"

# ================== Cargar fichero ==================
# Cambia aquí si quieres otro log:
# FILE_PATH = "Resultados/Trading_Log_AllStocks.csv"
FILE_PATH = "Resultados/Trading_Log_AllStocks_TARGET_TREND_ANG_15_5.csv"
# FILE_PATH = "Resultados/Trading_Log_ARIMA_AllStocks.csv"
# FILE_PATH = "Resultados/Trading_Log_AllStocks_MLP_OHLCV.csv"


if not Path(FILE_PATH).exists():
    raise FileNotFoundError(f"No se encuentra el fichero: {FILE_PATH}")

df = pd.read_csv(FILE_PATH, sep=';', encoding='utf-8')
if 'Fecha' not in df.columns or 'Symbol' not in df.columns:
    raise ValueError("El log debe contener columnas 'Fecha' y 'Symbol'.")

df['Fecha'] = pd.to_datetime(df['Fecha'])
df['Anio']  = df['Fecha'].dt.year

# Orden estable
orden_cols = ['Symbol', 'Fecha']
if 'OrdenN' in df.columns:
    orden_cols.append('OrdenN')
df.sort_values(orden_cols, inplace=True)
df.reset_index(drop=True, inplace=True)

# ================== Curva de capital por símbolo y cartera ==================
if 'Capital_Actual' not in df.columns:
    raise ValueError("Falta columna 'Capital_Actual' en el log.")

# Último registro por símbolo y día
df_daylast = df.groupby(['Symbol', 'Fecha'], as_index=False).tail(1)

# Pivot (Fecha x Symbol) con capital
pivot_cap = df_daylast.pivot_table(index='Fecha', columns='Symbol',
                                   values='Capital_Actual', aggfunc='last').sort_index()

# Completar continuidad por símbolo (si un día no operó)
pivot_cap = pivot_cap.bfill().ffill()

# Asegurar arranque normalizado a 10k por símbolo en el PRIMER año
years_sorted = sorted(pivot_cap.index.year.unique())
if not years_sorted:
    raise ValueError("No hay años en el dataset.")
first_year = years_sorted[0]
mask_first_year = pivot_cap.index.year == first_year
if mask_first_year.any():
    first_idx = pivot_cap.index[mask_first_year][0]
    for sym in pivot_cap.columns:
        if pd.notna(pivot_cap.loc[first_idx, sym]):
            pivot_cap.loc[first_idx, sym] = 10000.0
    pivot_cap.loc[mask_first_year] = pivot_cap.loc[mask_first_year].ffill()

# Curva de cartera (suma horizontal)
cartera_curve = pd.DataFrame({
    'Fecha': pivot_cap.index,
    'Capital_Cartera': pivot_cap.sum(axis=1)
})
cartera_curve['Anio'] = cartera_curve['Fecha'].dt.year

symbols = list(pivot_cap.columns)

# ================== Drawdown helper ==================
def max_drawdown(curva_capital):
    curva = np.asarray(curva_capital, dtype=float)
    if curva.size == 0:
        return 0.0
    running_max = np.maximum.accumulate(curva)
    drawdowns   = running_max - curva
    return float(np.nanmax(drawdowns))

# ================== Métricas POR STOCK y AÑO ==================
# Capital inicial/final por símbolo y año desde la curva por símbolo
rows_cap = []
for sym in symbols:
    serie = pivot_cap[sym].dropna()
    if serie.empty:
        continue
    tmp = pd.DataFrame({'Fecha': serie.index, 'Capital': serie.values})
    tmp['Anio'] = tmp['Fecha'].dt.year
    g = tmp.groupby('Anio')['Capital']
    for anio, s in g:
        rows_cap.append({
            'Symbol': sym,
            'Anio': int(anio),
            'Capital inicial': float(s.iloc[0]),
            'Capital final':   float(s.iloc[-1])
        })
cap_symbol_year = pd.DataFrame(rows_cap)

# Métricas de trading por símbolo/año (puede haber años sin trades)
is_sell = df['Accion'].astype(str).str.startswith('SELL', na=False) if 'Accion' in df.columns else pd.Series(False, index=df.index)
df_trades = df[is_sell].copy() if 'Profit' in df.columns else pd.DataFrame(columns=df.columns)

rows_trade = []
if not df_trades.empty:
    for (sym, anio), g in df_trades.groupby(['Symbol', 'Anio']):
        n_trades = int(g.shape[0])
        n_win    = int((g['Profit'] > 0).sum()) if 'Profit' in g.columns else 0
        win_rate = (n_win / n_trades * 100.0) if n_trades > 0 else 0.0
        avg_gain = g.loc[g['Profit'] > 0, 'Profit'].mean() if 'Profit' in g.columns else np.nan
        avg_loss = g.loc[g['Profit'] < 0, 'Profit'].mean() if 'Profit' in g.columns else np.nan
        payoff   = (avg_gain / abs(avg_loss)) if pd.notna(avg_gain) and pd.notna(avg_loss) and avg_loss < 0 else np.nan
        expectancy = (win_rate/100.0) * (avg_gain if pd.notna(avg_gain) else 0.0) + \
                     (1 - win_rate/100.0) * (avg_loss if pd.notna(avg_loss) else 0.0)
        rows_trade.append({
            'Symbol': sym, 'Anio': int(anio),
            'Trades': n_trades,
            'Ganadores (%)': win_rate,
            'Payoff Ratio': payoff,
            'Expectancy ($)': expectancy
        })

trades_metrics = pd.DataFrame(rows_trade) if rows_trade else pd.DataFrame(columns=[
    'Symbol','Anio','Trades','Ganadores (%)','Payoff Ratio','Expectancy ($)'
])

# Unir capitales y métricas
df_stock_year = pd.merge(cap_symbol_year, trades_metrics, on=['Symbol','Anio'], how='left')

# Rellenos y tipos seguros (años sin trades)
df_stock_year['Trades']         = pd.to_numeric(df_stock_year['Trades'], errors='coerce').fillna(0).astype(int)
df_stock_year['Ganadores (%)']  = pd.to_numeric(df_stock_year['Ganadores (%)'], errors='coerce')
df_stock_year['Payoff Ratio']   = pd.to_numeric(df_stock_year['Payoff Ratio'], errors='coerce')
df_stock_year['Expectancy ($)'] = pd.to_numeric(df_stock_year['Expectancy ($)'], errors='coerce')

df_stock_year['Ganancia ($)']  = df_stock_year['Capital final'] - df_stock_year['Capital inicial']
df_stock_year['ROI anual (%)'] = 100.0 * df_stock_year['Ganancia ($)'] / df_stock_year['Capital inicial']

# ================== Print: MÉTRICAS POR STOCK ==================
print("===== MÉTRICAS ANUALIZADAS POR STOCK =====")
for _, res in df_stock_year.sort_values(['Anio','Symbol']).iterrows():
    print(
        f"{int(res['Anio'])} | {res['Symbol']:>6} | "
        f"Init: {formato_espanol_numero(res['Capital inicial']):>12} | "
        f"Fin: {formato_espanol_numero(res['Capital final']):>12} | "
        f"ROI: {formato_espanol_numero(res['ROI anual (%)'], porcentaje=True):>8} | "
        f"Trades: {res['Trades']:>3} | "
        f"WinRate: {formato_espanol_numero(res['Ganadores (%)'], decimales=1, porcentaje=True):>7} | "
        f"Payoff: {formato_espanol_numero(res['Payoff Ratio']):>5} | "
        f"Exp.: {formato_espanol_numero(res['Expectancy ($)']):>10}"
    )

# ================== MÉTRICAS DE LA CARTERA (agregado por año) ==================
cartera_anual = cartera_curve.groupby('Anio').agg(
    Capital_inicial=('Capital_Cartera', 'first'),
    Capital_final=('Capital_Cartera', 'last')
).reset_index().sort_values('Anio')

# Forzar continuidad explícita
for i in range(1, len(cartera_anual)):
    cartera_anual.loc[i, 'Capital_inicial'] = cartera_anual.loc[i-1, 'Capital_final']

# Primer año = 10k * nº símbolos
if len(cartera_anual) > 0:
    cartera_anual.loc[0, 'Capital_inicial'] = 10000.0 * len(symbols)

cartera_anual['Ganancia ($)']  = cartera_anual['Capital_final'] - cartera_anual['Capital_inicial']
cartera_anual['ROI anual (%)'] = 100.0 * cartera_anual['Ganancia ($)'] / cartera_anual['Capital_inicial']

# Drawdown real anual sobre curva agregada
dd_anual = cartera_curve.groupby('Anio')['Capital_Cartera'].apply(max_drawdown).round(2)
cartera_anual['Drawdown máx ($)'] = cartera_anual['Anio'].map(dd_anual)

# Agregados ponderados por nº de trades
tmp = df_stock_year.copy()
tmp['Trades'] = pd.to_numeric(tmp['Trades'], errors='coerce').fillna(0).astype(int)
tmp['Payoff Ratio'] = pd.to_numeric(tmp['Payoff Ratio'], errors='coerce')

agg_rows = []
for anio, g in tmp.groupby('Anio'):
    trades_sum = int(g['Trades'].sum())
    if trades_sum > 0:
        wr = np.average(g['Ganadores (%)'].fillna(0), weights=g['Trades'])
        exp = np.average(g['Expectancy ($)'].fillna(0), weights=g['Trades'])
        mask_p = g['Payoff Ratio'].notna()
        payoff = np.average(g.loc[mask_p, 'Payoff Ratio'], weights=g.loc[mask_p, 'Trades']) if mask_p.any() else np.nan
    else:
        wr, exp, payoff = np.nan, np.nan, np.nan
    agg_rows.append({'Anio': int(anio), 'Trades': trades_sum,
                     'Ganadores (%)': wr, 'Payoff Ratio': payoff, 'Expectancy ($)': exp})

agg_df = pd.DataFrame(agg_rows)
cartera_anual = pd.merge(cartera_anual, agg_df, on='Anio', how='left')

print("\n===== MÉTRICAS DE LA CARTERA (SUMA DE ACTIVOS) POR AÑO =====")
print(cartera_anual.to_string(
    index=False,
    formatters={
        'Capital_inicial':     lambda x: formato_espanol_numero(x),
        'Capital_final':       lambda x: formato_espanol_numero(x),
        'Ganancia ($)':        lambda x: formato_espanol_numero(x),
        'ROI anual (%)':       lambda x: formato_espanol_numero(x, porcentaje=True),
        'Drawdown máx ($)':    lambda x: formato_espanol_numero(x),
        'Trades':              lambda x: f"{int(x):,}".replace(',', '.'),
        'Ganadores (%)':       lambda x: formato_espanol_numero(x, decimales=1, porcentaje=True) if np.isfinite(x) else '-',
        'Payoff Ratio':        lambda x: formato_espanol_numero(x) if np.isfinite(x) else '-',
        'Expectancy ($)':      lambda x: formato_espanol_numero(x) if np.isfinite(x) else '-',
    }
))

# ================== CAGR (cartera completa) ==================
first_date = cartera_curve['Fecha'].min()
last_date  = cartera_curve['Fecha'].max()
capital_inicio_total = float(cartera_curve.loc[cartera_curve['Fecha'] == first_date, 'Capital_Cartera'].iloc[0])
capital_fin_total    = float(cartera_curve.loc[cartera_curve['Fecha'] == last_date,  'Capital_Cartera'].iloc[0])

n_years_exact = max((last_date - first_date).days / 365.25, 1e-9)
initial_norm = 10000.0 * len(symbols)
CAGR = (capital_fin_total / initial_norm) ** (1.0 / n_years_exact) - 1.0 if initial_norm > 0 else np.nan

print(f"\nCAGR de la cartera (normalizada a 10k por símbolo): {formato_espanol_numero(CAGR*100, decimales=2)}%")
print("\n===== RESUMEN OK =====")
