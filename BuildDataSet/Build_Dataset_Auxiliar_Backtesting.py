import pandas as pd

# Cargar histórico completo
df_hist = pd.read_csv("DATA/AllStocksHistoricalData.csv", sep=';', parse_dates=['Fecha'])
df_hist = df_hist.sort_values(['Symbol', 'Fecha']).reset_index(drop=True)

def calc_atr_simple(df_symbol, window=8):
    high = df_symbol['High']
    low = df_symbol['Low']
    close = df_symbol['Close']
    prev_close = close.shift(1)

    tr = pd.DataFrame({
        'HL': high - low,
        'HC': (high - prev_close).abs(),
        'LC': (low - prev_close).abs()
    })
    tr_max = tr.max(axis=1)
    atr = tr_max.ewm(span=window, min_periods=window).mean()
    return atr

# Lista para acumular los DataFrames con ATR ya calculado
df_list = []

for symbol in df_hist['Symbol'].unique():
    df_symbol = df_hist[df_hist['Symbol'] == symbol].copy()
    df_symbol['ATR_8'] = calc_atr_simple(df_symbol, window=8).values
    df_list.append(df_symbol)

# Unir todo en un solo DataFrame final
df_atr = pd.concat(df_list, ignore_index=True)

# Guardar el resultado
df_atr.to_csv("DATA/AllStocksHistoricalData_Auxiliar_Backtesting.csv", sep=';', index=False)