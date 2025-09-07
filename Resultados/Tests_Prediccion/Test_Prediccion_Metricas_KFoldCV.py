import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.linear_model import ElasticNet, Ridge, Lasso, LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn

"""
Devuelve una instancia del modelo solicitado.
Para 'MLP-Torch', devuelve la arquitectura lista para entrenar.
"""
def get_model_by_name(model_name, input_dim=None):
    if model_name == "XGBRegressor":
        return XGBRegressor(n_estimators=60, max_depth=4, random_state=42, n_jobs=32)
    
    elif model_name == "ElasticNet":
        return ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=1000, random_state=42)
    
    elif model_name == "Ridge":
        return Ridge(alpha=1.0, random_state=42)
    
    elif model_name == "Lasso":
        return Lasso(alpha=0.1, random_state=42)
    
    elif model_name == "LinearRegression":
        return LinearRegression()
    
    elif model_name == "SVR":
        return SVR(kernel='rbf', C=1.0, epsilon=0.1)
    
    elif model_name == "MLPRegressor":
        return MLPRegressor(
            hidden_layer_sizes=(32,), activation='logistic', solver='adam',
            max_iter=500, early_stopping=True, n_iter_no_change=10, learning_rate='adaptive', alpha=0.001,
            tol=1e-4, random_state=42, verbose=False
        )
    
    elif model_name == "MLP-Torch":
        # Arquitectura simple idéntica al GA
        class SimpleMLP(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, 32),
                    nn.Sigmoid(),
                    nn.Dropout(0.2),
                    nn.Linear(32, 1),
                    nn.Sigmoid()
                )
            def forward(self, x):
                return self.net(x)
        return SimpleMLP(input_dim)
    else:
        raise ValueError("Modelo no soportado: " + model_name)

def cross_val_scores_symbol_years(df, symbol, features, target_col, years_test, model_name):
    """
    Calcula métricas de validación cruzada temporal para un símbolo concreto,
    usando años predefinidos como folds y el modelo especificado.
    Si se usa 'MLP-Torch', inicializa arquitectura y GPU igual que en el GA.
    """
    df_symbol = df[df['Symbol'] == symbol].sort_values('Fecha')
    scores = []         # Métricas por fold
    predicciones = []   # Predicciones fold a fold

    for year in years_test:
        
        # Definimos train/test conforme a los años de CV
        df_train = df[df['Fecha'].dt.year < year]
        df_test = df_symbol[df_symbol['Fecha'].dt.year == year]

        if df_test.empty or df_train.empty:
            continue

        # escogemos las features y la columna objetivo para train y test
        X_train = df_train[features]
        y_train = df_train[target_col]
        X_test = df_test[features]
        y_test = df_test[target_col]

        print(f"Comienza Test Out Of Sample para stock {symbol} año {year} y modelo {model_name}. Filas de train: {len(X_train)}, test: {len(X_test)}")

        # --- Entrenamiento y predicción ---
        if model_name == "MLP-Torch":
            # --- Inicialización e info GPU igual que en GA ---
            print(f"torch.cuda.is_available = {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print("Dispositivos CUDA disponibles:")
                print(torch.cuda.device_count())
                print("Dispositivo actual:", torch.cuda.current_device())
                print("Nombre:", torch.cuda.get_device_name(torch.cuda.current_device()))
            torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            input_dim = len(features)
            torch_mlp = get_model_by_name("MLP-Torch", input_dim=input_dim).to(torch_device)

            # Convertimos los datos a tensores
            X_train_t = torch.tensor(X_train.values, dtype=torch.float32, device=torch_device)
            y_train_t = torch.tensor(y_train.values, dtype=torch.float32, device=torch_device).view(-1, 1)
            X_test_t = torch.tensor(X_test.values, dtype=torch.float32, device=torch_device)

            optimizer = torch.optim.Adam(torch_mlp.parameters(), lr=0.001, weight_decay=0.0003)
            loss_fn = nn.MSELoss()

            best_loss = float('inf')
            epochs_no_improve = 10
            n_iter_no_change = 10
            best_state = None

            for epoch in range(500):
                torch_mlp.train()
                optimizer.zero_grad()
                output = torch_mlp(X_train_t)
                loss = loss_fn(output, y_train_t)
                loss.backward()
                optimizer.step()
                current_loss = loss.item()
                if current_loss < best_loss - 1e-4:
                    best_loss = current_loss
                    best_state = torch_mlp.state_dict()
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= n_iter_no_change:
                        print(f"Early stopping at epoch {epoch} con loss={best_loss:.6f}")
                        break
            if best_state is not None:
                torch_mlp.load_state_dict(best_state)
            torch_mlp.eval()
            with torch.no_grad():
                y_pred = torch_mlp(X_test_t).cpu().numpy().flatten()
        else:
            
            # Modelos clásicos sklearn/xgboost
            model = get_model_by_name(model_name, input_dim=len(features))
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        
        # --- Métricas por fold ---
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        scores.append({
            'year': year,
            'train_hasta_año': year - 1,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'n_train': len(X_train),
            'n_test': len(X_test)
        })

        # --- Guardamos predicciones fold a fold para análisis posterior ---
        df_preds = pd.DataFrame({
            'Fecha': df_test['Fecha'].values,
            'Symbol': symbol,
            'Fold': year,
            'Target': y_test.values,
            'Predicted': y_pred,
            'Model': model_name,
            'TargetName': target_col
        })
        predicciones.append(df_preds)
    
    return scores, predicciones

def main():

    # --- CONFIGURACIÓN PRINCIPAL ---
    # Modelo a usar: XGBRegressor, ElasticNet, Ridge, Lasso, LinearRegression, SVR, MLPRegressor, MLP-Torch
    model_name = "MLPRegressor"  

    # Cargamos dataset y preparamos features
    df = pd.read_csv("DATA/Dataset_All_Features_Transformado.csv", sep=";", parse_dates=["Fecha"])
    
    # Excluimos las columnas que no son features
    cols_a_excluir = ['Fecha', 'Symbol', 'Open', 'Close', 'High', 'Low', 'AdjClose', 'Volume']
    cols_a_excluir += [c for c in df.columns if c.startswith('EMA_')]
    features = [c for c in df.columns if c not in cols_a_excluir and not c.startswith('TARGET_')]

    # Si utilizamos GA, podemos especificar aquí  las features seleccionadas con GA
    # features =  ['RSI_10', 'CCI_12', ...]  # Ejemplo de features seleccionadas
     

    # Columna objetivo
    target_col = "TARGET_TREND_ANG_15_5"
    
    # Años de testear (K-Fold CV)
    years_test = [2020, 2021, 2022, 2023, 2024]
    
    # Activo financiero
    symbol = "NVDA"

    print(f"\nResultados para {symbol} utilizando modelo {model_name} con {len(features)} features:")

    try:
        # Ejecutamos validación cruzada temporal para el símbolo y modelo especificado
        scores, predicciones = cross_val_scores_symbol_years(
            df, symbol, features, target_col, years_test, model_name)

    except ValueError as e:
        print(str(e))
        return


    # Imprimimos métricas fold a fold
    for s in scores:
        print(f"Fold año {s['year']}: train_hasta_año={s['train_hasta_año']} | "
              f"n_train={s['n_train']}, n_test={s['n_test']}, "
              f"RMSE={s['rmse']:.4f}, R2={s['r2']:.3f}")

    resumen = {k: np.mean([s[k] for s in scores]) for k in ['mse','rmse','mae','r2']}
    resumen['Symbol'] = symbol
    print("\nResumen promedio para", symbol)

    # Imprimimos el resumen de métricas
    for k in ['mse','rmse','mae','r2']:
        print(f"{k.upper()}: {resumen[k]:.4f}")

    
    # Guardamos las métricas por fold en un DataFrame
    df_all_preds = pd.concat(predicciones, ignore_index=True)
    output_path = f"Resultados/Predicciones_KFoldsCV_{model_name}.csv"
    
    # Guardamos el DataFrame con las predicciones
    df_all_preds.to_csv(output_path, index=False)
    print(f"\nArchivo CSV con predicciones guardado en '{output_path}'")

if __name__ == "__main__":
    main()
