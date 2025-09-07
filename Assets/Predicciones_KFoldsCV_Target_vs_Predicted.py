import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURACIÓN ---
csv_path = "Resultados/Predicciones_KFoldsCV_XGBRegressor.csv"
years_plot = [2022, 2023]  # <-- Cambia aquí los años a visualizar

# --- Carga y preprocesa ---
df = pd.read_csv(csv_path, parse_dates=["Fecha"])
df['Year'] = pd.to_datetime(df['Fecha']).dt.year
df_plot = df[df['Year'].isin(years_plot)].copy()
df_plot = df_plot.sort_values("Fecha")

# --- Set estilo Seaborn pro ---
sns.set_theme(style="whitegrid")
plt.figure(figsize=(16, 7))
plt.rcParams.update({'font.size': 13})

# --- Línea Target (Real): sólida azul navy ---
sns.lineplot(
    x="Fecha", y="Target", data=df_plot,
    label="Target (Real)", color="navy",
    linewidth=2.4, alpha=0.92, linestyle="-"
)

# --- Línea Predicted: discontinua naranja encima ---
sns.lineplot(
    x="Fecha", y="Predicted", data=df_plot,
    label="Predicted (Modelo)", color="darkorange",
    linewidth=2, alpha=0.85, linestyle="--"
)

# --- Leyenda y detalles ---
plt.title(
    f"Target vs Predicted (NVDA {min(years_plot)}-{max(years_plot)}) - Modelo: {df_plot['Model'].iloc[0]}",
    fontsize=17, fontweight="bold"
)
plt.xlabel("Fecha", fontweight="bold")
plt.ylabel(df_plot["TargetName"].iloc[0], fontweight="bold")
plt.legend(fontsize=13, loc="upper right", frameon=True)
plt.xticks(rotation=25)
plt.tight_layout()
plt.grid(True, which="both", axis="y", alpha=0.18)

# --- Save pro para el TFM ---
# plt.savefig(f"Resultados/TFM_Target_vs_Predicted_{min(years_plot)}_{max(years_plot)}.png", dpi=250)
plt.show()
