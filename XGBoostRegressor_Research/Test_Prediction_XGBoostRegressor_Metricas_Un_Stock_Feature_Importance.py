import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import shap

# 1. Cargar el dataset
df = pd.read_csv("DATA/Dataset_All_Features_Transformado.csv", parse_dates=['Fecha'], sep=';')


# 2. Definir las columnas a usar como features
cols_a_excluir = ['Fecha', 'Symbol', 'Open', 'Close', 'High', 'Low', 'AdjClose', 'Volume']
cols_a_excluir += [col for col in df.columns if col.startswith('EMA_')]  # EMAs absolutos

# 3. Qutamos las columnas que no son features
# (por ejemplo, TARGET_SLOPE_10, TARGET_SLOPE_20, etc.)
features = [ c for c in df.columns   if c not in cols_a_excluir   and not c.startswith('TARGET_')]


# 4. Seleccionamos el target
target = 'TARGET_TREND_ANG_15_5'


# 5. Ordenar por símbolo y fecha
df = df.sort_values(['Symbol', 'Fecha']).reset_index(drop=True)

# 6. Definir símbolo a testear
symbol_test = 'NVDA'  # Cambia aquí el símbolo que quieres predecir

# 7. Calcula split_date global (usando percentil 80% de la columna Fecha del primer símbolo)
primer_symbol = df['Symbol'].unique()[0]
fechas_primer_symbol = df[df['Symbol'] == primer_symbol]['Fecha']

split_index = int(len(fechas_primer_symbol) * 0.8)
split_date = fechas_primer_symbol.iloc[split_index]

print(f"Fecha de split global: {split_date}")

# 8. Crear train/test: entrena con todos menos el de test, predice TODO el símbolo de test después de split_date
df_train = df[df['Symbol'] != symbol_test]
df_test_full = df[df['Symbol'] == symbol_test].sort_values('Fecha')
df_test = df_test_full[df_test_full['Fecha'] > split_date]  # Opcional: solo fechas posteriores

X_train = df_train[features]
y_train = df_train[target]
X_test = df_test[features]
y_test = df_test[target]

print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

# 9. Normalizar: ajusta el scaler SOLO con train, luego transforma test
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 10. Entrenar modelo
model = XGBRegressor(n_estimators=120, max_depth=4, random_state=42, n_jobs=30)
model.fit(X_train_scaled, y_train)

# 11. Predicciones y métricas
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"R2: {r2:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")




# Si features está definido, úsalo. Si no, haz una lista de columnas de X_train.
# Aquí 'features' es tu lista de nombres de columnas usadas en el modelo
importances = model.feature_importances_
feature_names = X_train.columns

# Ordenar de mayor a menor
indices = np.argsort(importances)[::-1]
top_n = 20  # Cambia este valor si quieres ver más o menos features

plt.figure(figsize=(12, 8))
plt.title("Top 50 Feature Importances (XGBoost)")
plt.bar(range(top_n), importances[indices][:top_n], align="center")
plt.xticks(range(top_n), [feature_names[i] for i in indices[:top_n]], rotation=90)
plt.tight_layout()
plt.show()

#Opcional: imprimir ranking en consola
for i in range(top_n):
   print(f"{i+1:2d}. {feature_names[indices[i]]}: {importances[indices[i]]:.5f}")




# Creamos el explainer SHAP para XGBoost
explainer = shap.Explainer(model, X_train)

# Calcula los valores SHAP para el test set
shap_values = explainer(X_test)

# Resumen global: gráfico de barras (importancia media absoluta)
shap.summary_plot(shap_values, X_test, max_display=top_n)