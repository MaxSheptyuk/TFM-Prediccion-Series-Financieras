import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.linear_model import ElasticNet, Ridge, Lasso, LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn

def get_model_by_name(model_name, input_dim=None):
    if model_name == "XGBRegressor":
        return XGBRegressor(n_estimators=60, max_depth=4, random_state=42, n_jobs=30)
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
            hidden_layer_sizes=(16,), activation='relu', solver='adam',
            max_iter=200, early_stopping=True, n_iter_no_change=5,
            tol=1e-3, random_state=42, verbose=False
        )
    elif model_name == "MLP-Torch":
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

def train_and_predict_once(df, symbol, features, target_col, train_years, test_years, model_name):
    # Filtra solo el símbolo seleccionado
    df_symbol = df[df['Symbol'] == symbol].sort_values('Fecha')

    # Split train/test
    df_train = df[df['Fecha'].dt.year.isin(train_years)]
    df_test = df_symbol[df_symbol['Fecha'].dt.year.isin(test_years)]

    if df_test.empty or df_train.empty:
        raise ValueError("Train o Test vacío para el rango de años indicado.")

    X_train = df_train[features]
    y_train = df_train[target_col]
    X_test = df_test[features]
    y_test = df_test[target_col]

    print(f"Entrenando {model_name} con {len(X_train)} muestras de train. Test: {len(X_test)} muestras.")

    # --- Entrenamiento y predicción ---
    if model_name == "MLP-Torch":
        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_dim = len(features)
        torch_mlp = get_model_by_name("MLP-Torch", input_dim=input_dim).to(torch_device)
        X_train_t = torch.tensor(X_train.values, dtype=torch.float32, device=torch_device)
        y_train_t = torch.tensor(y_train.values, dtype=torch.float32, device=torch_device).view(-1, 1)
        X_test_t = torch.tensor(X_test.values, dtype=torch.float32, device=torch_device)
        optimizer = torch.optim.Adam(torch_mlp.parameters(), lr=0.001, weight_decay=0.0003)
        loss_fn = nn.MSELoss()
        best_loss = float('inf')
        epochs_no_improve = 0
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
        model = get_model_by_name(model_name, input_dim=len(features))
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    # Métricas de test
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"\nTest - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

    # Prepara el DataFrame de salida
    df_preds = pd.DataFrame({
        'Fecha': df_test['Fecha'].values,
        'Stock': symbol,
        'Target': y_test.values,
        'Predicted': y_pred,
        'Model': model_name
    })
    return df_preds

def main():
    # --- Configuración principal ---
    model_name = "XGBRegressor"   # Cambia aquí el modelo
    df = pd.read_csv("DATA/Dataset_All_Features_Transformado.csv", sep=";", parse_dates=["Fecha"])

    # Features y target
    cols_a_excluir = ['Fecha', 'Symbol', 'Open', 'Close', 'High', 'Low', 'AdjClose', 'Volume']
    cols_a_excluir += [c for c in df.columns if c.startswith('EMA_')]
    features = [c for c in df.columns if c not in cols_a_excluir and not c.startswith('TARGET_')]

    target_col = "TARGET_TREND_ANG_15_5"

    # Rango de años de train y test
    train_years = [year for year in range(2010, 2021)]   # Ejemplo: 2010-2020
    test_years = [2021, 2022, 2023, 2024]

    symbol = "NVDA"

    print(f"Entrenando y testando para {symbol} y modelo {model_name}...")
    df_preds = train_and_predict_once(
        df, symbol, features, target_col, train_years, test_years, model_name
    )

    # Guarda el CSV con columnas fecha, stock, target, predicted, model
    output_path = f"Resultados/Predicciones_Test_{model_name}_{symbol}.csv"
    df_preds.to_csv(output_path, index=False)
    print(f"\nPredicciones guardadas en '{output_path}'")

if __name__ == "__main__":
    main()
