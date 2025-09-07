import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def cross_val_scores_symbol_years(df, symbol, features, target_col, years_test):

    df_symbol = df[df['Symbol'] == symbol].sort_values('Fecha')
    scores = []

    for year in years_test:

        df_train = df[df['Fecha'].dt.year < year]
        df_test = df_symbol[df_symbol['Fecha'].dt.year == year]

        if df_test.empty or df_train.empty:
            continue

        X_train = df_train[features]
        y_train = df_train[target_col]
        X_test = df_test[features]
        y_test = df_test[target_col]

        model = XGBRegressor(n_estimators=60, max_depth=4, random_state=42, n_jobs=30)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        scores.append({
            'Symbol': symbol,
            'year': year,
            'train_hasta_año': year - 1,
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'n_train': len(X_train),
            'n_test': len(X_test)
        })
    return scores

def main():
    # Cargamos el dataset ya preprocesado, asegurando que la columna Fecha es datetime
    df = pd.read_csv("DATA/Dataset_All_Features_Transformado.csv", sep=";", parse_dates=["Fecha"])
    
    # Excluimos columnas que no son features
    cols_a_excluir = ['Fecha', 'Symbol', 'Open', 'Close', 'High', 'Low', 'AdjClose', 'Volume']
    cols_a_excluir += [c for c in df.columns if c.startswith('EMA_')]
    features = [c for c in df.columns if c not in cols_a_excluir and not c.startswith('TARGET_')]
    target_col = "TARGET_TREND_ANG_15_5"
    years_test = [2020, 2021, 2022, 2023, 2024]

    all_scores = []
    resumen_promedio = []

    for symbol in sorted(df['Symbol'].unique()):
        print(f"\nProcesando {symbol}...")
        scores = cross_val_scores_symbol_years(df, symbol, features, target_col, years_test)
        if not scores:
            print(f"  Sin datos de test suficientes para {symbol}")
            continue

        all_scores.extend(scores)

        resumen = {
            'Symbol': symbol,
            'mse': float(np.mean([s['mse'] for s in scores])),
            'rmse': float(np.mean([s['rmse'] for s in scores])),
            'mae': float(np.mean([s['mae'] for s in scores])),
            'r2': float(np.mean([s['r2'] for s in scores])),
        }
        print(f"Resumen promedio para {symbol}:  "
              f"mse={resumen['mse']:.6f}, rmse={resumen['rmse']:.6f}, "
              f"mae={resumen['mae']:.6f}, r2={resumen['r2']:.6f}")
        resumen_promedio.append(resumen)

    df_scores = pd.DataFrame(all_scores)
    df_resumen = pd.DataFrame(resumen_promedio)[['Symbol', 'mse', 'rmse', 'mae', 'r2']]
    df_scores.to_csv("RESULTADOS/Metricas_XGBoostRegressor_CV_Completas_por_Fold.csv", index=False)
    df_resumen.to_csv("RESULTADOS/Metricas_XGBoostRegressor_Promedio_Por_Symbol_CV.csv", index=False)

    print("\nArchivos guardados:")
    print(" - Metricas_XGBoostRegressor_CV_Completas_por_Fold.csv")
    print(" - Metricas_XGBoostRegressor_Promedio_Por_Symbol_CV.csv")

if __name__ == "__main__":
    main()
