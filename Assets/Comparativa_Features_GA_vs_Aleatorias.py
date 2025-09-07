import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Carga tu nuevo fichero de resultados
df = pd.read_csv("Resultados/Comparativa_GA_vs_Aleatorias.csv")
palette = {"GA": "#F28E2B", "Aleatorias": "#4E79A7"}
sns.set_theme(style="whitegrid", font_scale=1.13)

def plot_comparativa(df, metrica="R2", titulo=None):
    """
    Gráfico de barras para comparar GA vs Aleatorias en la métrica seleccionada (por stock).
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(
        data=df, x="Stock", y=metrica, hue="Metodo",
        palette=palette, edgecolor="black", linewidth=1.1,
        dodge=True, width=0.5, ax=ax
    )
    ax.set_title(titulo or f"{metrica} por activo: GA vs Aleatorias", fontsize=15, pad=13)
    ylabel = {
        "R2": "R² (Coef. determinación)",
        "RMSE": "RMSE (Error cuadrático medio)",
        "MAE": "MAE (Error absoluto medio)"
    }.get(metrica, metrica)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel("Activo Financiero", fontsize=12, labelpad=20)
    plt.xticks(rotation=28, ha="right")
    leg = ax.legend(
        title="Método", loc='center left', bbox_to_anchor=(1.01, 0.5),
        frameon=True, fontsize=11, title_fontsize=12
    )
    leg.get_frame().set_alpha(0.95)
    ax.grid(axis="y", linestyle="--", alpha=0.53)
    plt.tight_layout(rect=[0, 0, 0.98, 1])
    plt.show()

# Ejemplo: para RMSE (puedes cambiar por "MAE" o "R2")
plot_comparativa(df, "RMSE", "RMSE medio (3 K-Fold CV) por activo financiero: GA vs Aleatorias")

# Para R2
#plot_comparativa(df, "R2", "R² medio (3 K-Fold CV) por activo: GA vs Aleatorias")

# Para MAE
#plot_comparativa(df, "MAE", "MAE medio (3 K-Fold CV) por activo: GA vs Aleatorias")
