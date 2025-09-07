import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import shap

# --- Carga dataset ---
df = pd.read_csv("DATA/Dataset_All_Features_Transformado.csv", sep=";", parse_dates=["Fecha"])


# --- Definir features y target ---
cols_a_excluir = ['Fecha', 'Symbol', 'Open', 'Close', 'High', 'Low', 'AdjClose', 'Volume']
cols_a_excluir += [c for c in df.columns if c.startswith('EMA_')]  # Excluir EMAs absolutos

features = [c for c in df.columns if c not in cols_a_excluir and not c.startswith('TARGET_')]


target_col = 'TARGET_TREND_ANG_15_5' 

# --- Ordenar dataset ---
df = df.sort_values(['Symbol', 'Fecha']).reset_index(drop=True)

# --- Elegir símbolo para test ---
symbol_test = 'NVDA'

# --- Fecha de split basada en percentil 80 del primer símbolo ---
primer_symbol = df['Symbol'].unique()[0]
fechas_primer_symbol = df[df['Symbol'] == primer_symbol]['Fecha']
split_index = int(len(fechas_primer_symbol) * 0.8)
split_date = fechas_primer_symbol.iloc[split_index]

print(f"Fecha de split: {split_date}")

# --- Crear train y test splits ---
df_train = df[(df['Symbol'] != symbol_test) & (df['Fecha'] <= split_date)]
df_test = df[(df['Symbol'] == symbol_test) & (df['Fecha'] > split_date)]

# --- Extraer X e y ---
X_train = df_train[features]
y_train = df_train[target_col]
X_test = df_test[features]
y_test = df_test[target_col]


# --- Escalar features ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Entrenar XGBoost ---
model = XGBRegressor(n_estimators=120, max_depth=4, random_state=42, n_jobs=30)
model.fit(X_train_scaled, y_train)

# --- Predecir ---
y_pred = model.predict(X_test_scaled)

# --- Métricas ---
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R2: {r2:.4f}")


# --- Importancia de features ---
importances = model.feature_importances_
feature_names = X_train.columns

# Ordenar las features por importancia descendente
indices = np.argsort(importances)[::-1]

print("\nTop 50 características importantes:")
for i in range(50):
    print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.5f}")

# # --- Visualización gráfica ---
plt.figure(figsize=(12, 6))
plt.title("Top 20 Importancia de Features (XGBoost)")
plt.bar(range(50), importances[indices][:50], align='center')
plt.xticks(range(50), [feature_names[i] for i in indices[:50]], rotation=90)
plt.tight_layout()
plt.show()



# --- Explicabilidad con SHAP ---

# Crea el explainer para XGBoost con datos de entrenamiento sin escalar (mejor interpretación)
explainer = shap.Explainer(model, X_train_scaled)

# Calcula valores SHAP para el conjunto de test
shap_values = explainer(X_test_scaled)

# Gráfico resumen global: importancia media absoluta (top 50 features)
shap.summary_plot(shap_values, X_test, feature_names=features, max_display=50)

# Opcional: gráfico de barras para la importancia media absoluta
shap.summary_plot(shap_values, X_test, feature_names=features, plot_type="bar", max_display=50)