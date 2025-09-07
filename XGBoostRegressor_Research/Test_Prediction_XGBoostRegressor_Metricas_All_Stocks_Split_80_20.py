import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def train_test_split_symbol(df, symbol, features, target_col, split_ratio=0.8):
    """
    Crea el train-test split para los 25 símbolos de acciones.:
    - Entrena con todos los datos excepto los del símbolo seleccionado
    - Train Test Split cronológico  80%  20% basado en la fecha para los 25 stocks
    - Test solo para el símbolo seleccionado y para fechas posteriores a split_date
    """
    df_symbol = df[df['Symbol'] == symbol].sort_values('Fecha')
    split_index = int(len(df_symbol) * split_ratio)
    split_date = df_symbol.iloc[split_index]['Fecha']
    
    # Train: todos los símbolos hasta el split_date
    # Incluye el símbolo seleccionado hasta la fecha de split
    df_train = df[df['Fecha'] <= split_date]

    # Test: solo filas del símbolo seleccionado con fecha posterior al split_date
    df_test = df_symbol[df_symbol['Fecha'] > split_date]
    
    X_train = df_train[features]
    y_train = df_train[target_col]
    X_test = df_test[features]
    y_test = df_test[target_col]
    
    return X_train, y_train, X_test, y_test, split_date

def train_and_evaluate(X_train, y_train, X_test, y_test):
    """
    Entrena el modelo y evalúa el rendimiento sobre el set de test.
    Devuelve el modelo entrenado, las predicciones y un diccionario con las métricas.
    """
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = XGBRegressor(n_estimators=120, max_depth=4, random_state=42, n_jobs=30)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, y_pred, {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}

def main():

    # 1. Cargar dataset
    df = pd.read_csv("DATA/Dataset_All_Features_Transformado.csv", sep=";", parse_dates=["Fecha"])
    
    
    # 3. Definir columnas a excluir para features
    cols_a_excluir = ['Fecha', 'Symbol', 'Open', 'Close', 'High', 'Low', 'AdjClose', 'Volume']
    cols_a_excluir += [c for c in df.columns if c.startswith('EMA_')]
    
    # 4. Definir lista de features (sin columnas excluidas ni targets)
    features = [c for c in df.columns if c not in cols_a_excluir and not c.startswith('TARGET_')]

    print(f"Features seleccionadas: {features}")
    
    # 5. Seleccionar el target 
    target_col = "TARGET_TREND_ANG_15_5"
    
    
    # 6. Evaluar por cada símbolo en dataset
    metrics = []
    for symbol in df['Symbol'].unique():
        print(f"Procesando símbolo: {symbol}")
        
        # 7. Split train-test cronológico para este símbolo
        X_train, y_train, X_test, y_test, split_date = train_test_split_symbol(df, symbol, features, target_col)
        
        # 8. Comprobar si hay datos para test después de la fecha de split
        if X_test.empty or y_test.empty:
            print(f"  - No hay datos de test para {symbol} después del {split_date.date()}. Saltando...")
            continue
        
        # 9. Entrenar y evaluar modelo XGBoost
        model, y_pred, scores = train_and_evaluate(X_train, y_train, X_test, y_test)
        
        print(f"  - Fecha de split: {split_date.date()}")
        print(f"  - RMSE: {scores['rmse']:.4f} | MAE: {scores['mae']:.4f} | R2: {scores['r2']:.4f}")
        
        metrics.append({'Symbol': symbol, **scores})
    
    # 10. Mostrar resumen de métricas
    metrics_df = pd.DataFrame(metrics)
    print("\nResumen de métricas por símbolo:")
    print(metrics_df)
    
    return metrics_df

if __name__ == "__main__":
    main()
