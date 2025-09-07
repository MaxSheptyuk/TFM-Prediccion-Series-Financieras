import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

 # Personalizamos aquí
SYMBOLS_A_TESTEAR = ['MSFT', 'AMZN', 'CAT'] 
TARGETS_A_PROBAR = [
    'TARGET_TREND_ANG_10_1',
    'TARGET_TREND_ANG_10_2',
    'TARGET_TREND_ANG_10_3',
    'TARGET_TREND_ANG_10_4',
    'TARGET_TREND_ANG_10_5',
    'TARGET_TREND_ANG_10_6',
    'TARGET_TREND_ANG_10_7',
    'TARGET_TREND_ANG_10_8',
    'TARGET_TREND_ANG_10_9',
    'TARGET_TREND_ANG_10_10',

    'TARGET_TREND_ANG_15_3',
    'TARGET_TREND_ANG_15_5',
    'TARGET_TREND_ANG_15_8',
    'TARGET_TREND_ANG_15_10',
    'TARGET_TREND_ANG_15_12',
    'TARGET_TREND_ANG_15_15',
]

def train_test_split_symbol(df, symbol, features, target_col, split_ratio=0.8):
    
    df_symbol = df[df['Symbol'] == symbol].sort_values('Fecha')
    split_index = int(len(df_symbol) * split_ratio)
    split_date = df_symbol.iloc[split_index]['Fecha']
    
    df_train = df[df['Fecha'] <= split_date]
    df_test = df_symbol[df_symbol['Fecha'] > split_date]
    X_train = df_train[features]
    y_train = df_train[target_col]
    X_test = df_test[features]
    y_test = df_test[target_col]
    return X_train, y_train, X_test, y_test, split_date

def train_and_evaluate(X_train, y_train, X_test, y_test):
    model = XGBRegressor(n_estimators=120, max_depth=4, random_state=42, n_jobs=30)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return model, y_pred, {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}

def main():
    df = pd.read_csv("DATA/Dataset_All_Features_Transformado.csv", sep=";", parse_dates=["Fecha"])
    
    # Corte inicial desde 2015 para reducir latencias excesivas de entrenamiento
    df = df[df['Fecha'] >= pd.Timestamp('2015-01-01')]
    
    cols_a_excluir = ['Fecha', 'Symbol', 'Open', 'Close', 'High', 'Low', 'AdjClose', 'Volume']
    cols_a_excluir += [c for c in df.columns if c.startswith('EMA_')]
    features = [c for c in df.columns if c not in cols_a_excluir and not c.startswith('TARGET_')]
    all_metrics = []

    for target_col in TARGETS_A_PROBAR:
        print(f"\n\n===> Evaluando TARGET: {target_col}\n")
        for symbol in SYMBOLS_A_TESTEAR:
            print(f"Procesando símbolo: {symbol}")
            if target_col not in df.columns:
                print(f"  - Columna {target_col} no encontrada. Saltando...")
                continue
            X_train, y_train, X_test, y_test, split_date = train_test_split_symbol(
                df, symbol, features, target_col)
            if X_test.empty or y_test.empty:
                print(f"  - No hay datos de test para {symbol} después del {split_date.date()}. Saltando...")
                continue
            model, y_pred, scores = train_and_evaluate(X_train, y_train, X_test, y_test)
            all_metrics.append({
                'Target': target_col,
                'Symbol': symbol,
                'SplitDate': split_date.date(),
                **scores
            })
            print(f"  - RMSE: {scores['rmse']:.4f} | MAE: {scores['mae']:.4f} | R2: {scores['r2']:.4f}")

    metrics_df = pd.DataFrame(all_metrics)
    print("\n=== Resumen total ===\n", metrics_df)
    metrics_df.to_csv("RESULTADOS/Resultados_Window_Horizon.csv", index=False)
    return metrics_df

if __name__ == "__main__":
    main()
