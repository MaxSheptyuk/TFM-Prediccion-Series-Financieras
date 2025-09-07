import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import shap

# --- Carga dataset actualizado ---
df = pd.read_csv("DATA/Dataset_All_Features_Transformado.csv", sep=";")
df['Fecha'] = pd.to_datetime(df['Fecha'])
df = df[df['Fecha'] >= pd.Timestamp('2015-01-01')]  # Filtrar desde 2015

# --- Definir features y target ---
cols_a_excluir = ['Fecha', 'Symbol', 'Open', 'Close', 'High', 'Low', 'AdjClose', 'Volume']
cols_a_excluir += [c for c in df.columns if c.startswith('EMA_')]
features = [c for c in df.columns if c not in cols_a_excluir and not c.startswith('TARGET_')]

target_col = 'TARGET_TREND_ANG_15_5'  # Ajusta aquí el target que quieras analizar

# --- Stocks a analizar ---
SYMBOLS_A_TESTEAR = ['AAPL', 'MSFT', 'NVDA', 'CAT', 'ADI']  # Cambia por los 5 que prefieras

# --- Guardar importancias de todos los stocks ---
importancias_all = []

for symbol_test in SYMBOLS_A_TESTEAR:
    # Split por fechas de ese stock
    fechas_symbol = df[df['Symbol'] == symbol_test]['Fecha']
    split_index = int(len(fechas_symbol) * 0.8)
    split_date = fechas_symbol.iloc[split_index]
    print(f"Symbol {symbol_test} - Split date: {split_date}")

    df_train = df[(df['Symbol'] != symbol_test) & (df['Fecha'] <= split_date)]
    df_test = df[(df['Symbol'] == symbol_test) & (df['Fecha'] > split_date)]

    X_train = df_train[features]
    y_train = df_train[target_col]
    X_test = df_test[features]
    y_test = df_test[target_col]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = XGBRegressor(n_estimators=120, max_depth=4, random_state=42, n_jobs=30)
    model.fit(X_train_scaled, y_train)

    # Guarda importancia
    importancias_all.append(model.feature_importances_)

    # --- SHAP solo para uno (ej. NVDA) ---
    if symbol_test == 'NVDA':
        explainer = shap.Explainer(model, X_train_scaled)
        shap_values = explainer(X_test_scaled)
        print(f"\nSHAP para {symbol_test}")
        shap.summary_plot(shap_values, X_test, feature_names=features, max_display=30)
        shap.summary_plot(shap_values, X_test, feature_names=features, plot_type="bar", max_display=30)

# --- DataFrame de importancias (n_stocks x n_features) ---
importancias_all = np.vstack(importancias_all)
df_importancias = pd.DataFrame(importancias_all, columns=features, index=SYMBOLS_A_TESTEAR)
importancia_media = df_importancias.mean(axis=0).sort_values(ascending=False)

# --- Gráfico de barras: top 30 features promedio ---
topN = 30
plt.figure(figsize=(12, 6))
plt.title(f"Top {topN} Importancia Promedio de Features (XGBoost, 5 stocks)")
plt.bar(range(topN), importancia_media.iloc[:topN], align='center')
plt.xticks(range(topN), importancia_media.index[:topN], rotation=90)
plt.tight_layout()
plt.show()

print("\nTop 30 features promedio:")
print(importancia_media.head(topN))

# Opcional: guardar a csv
importancia_media.to_csv("RESULTADOS/importancia_features_promedio.csv")
