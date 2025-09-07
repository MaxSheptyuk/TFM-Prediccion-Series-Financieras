import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Carga el fichero de resultados
df = pd.read_csv("Resultados/Comparativa_GA_Multimodelo_KFold.csv")
sns.set_theme(style="whitegrid", font_scale=1.13)

# Opcional: si tu columna de modelos tiene nombres largos, puedes renombrarlos a corto
modelo_map = {
    "XGBRegressor": "XGBoost",
    "ElasticNet": "ElasticNet",
    "LinearRegression": "LinearRegr.",
    "Ridge": "Ridge",
    "SVR": "SVR",
    "Lasso": "Lasso",
    "MLP-Torch": "MLP (32-1)"
}
df["Modelo"] = df["Modelo"].map(modelo_map).fillna(df["Modelo"])

# Si quieres mostrar solo un stock, por ejemplo "NVDA"
df_plot = df[df["Stock"] == "NVDA"]

def plot_kfold_por_modelo(df, metrica="R2", titulo=None):
    """
    Gráfico de barras: cada grupo es un modelo, cada barra un fold (año).
    """
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(
        data=df, x="Modelo", y=metrica, hue="Año_Test",
        palette="Set2", edgecolor="black", linewidth=1.15,
        dodge=True, width=0.65, ax=ax
    )
    ax.set_title(titulo or f"{metrica} por modelo (cada barra un fold/año)", fontsize=15, pad=13)
    ylabel = {
        "R2": "R² (Coef. determinación)",
        "RMSE": "RMSE (Error cuadrático medio)",
        "MAE": "MAE (Error absoluto medio)"
    }.get(metrica, metrica)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel("Modelo", fontsize=12, labelpad=20)
    plt.xticks(rotation=0, ha="center")
    leg = ax.legend(
        title="Fold/Año Test", loc='center left', bbox_to_anchor=(1.01, 0.5),
        frameon=True, fontsize=11, title_fontsize=12
    )
    leg.get_frame().set_alpha(0.95)
    ax.grid(axis="y", linestyle="--", alpha=0.53)
    plt.tight_layout(rect=[0, 0, 0.98, 1])
    plt.show()

# Ejemplo para R2
plot_kfold_por_modelo(df_plot, "R2", "R² por modelo para activo NVDA: barras = folds (años test)")

# Para RMSE:
#plot_kfold_por_modelo(df_plot, "RMSE", "RMSE por modelo para activo NVDA: barras = folds (años test)")

# Para MAE:
# plot_kfold_por_modelo(df_plot, "MAE", "MAE por modelo para activo NVDA: barras = folds (años test)")
