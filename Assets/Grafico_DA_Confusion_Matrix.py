# plot_confusion_grid_2col_labels_peraxis.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patheffects
from sklearn.metrics import confusion_matrix, classification_report  # <-- añadido

CSV_PATH = "Resultados/DA_Binary_Daily_AllStocks_SIMPLE.csv"

def _reconstruct_ytrue(df_sym: pd.DataFrame):
    y_pred = df_sym["y_pred_bin"].astype(int).to_numpy()
    correct = df_sym["correct"].astype(int).to_numpy()
    y_true = np.where(correct == 1, y_pred, 1 - y_pred)
    return y_true, y_pred

def plot_confusion_grid_2cols(csv_path: str,
                              symbols: list[str],
                              normalize: bool = False,
                              per_axis_scale: bool = True,   # cada subplot con su escala
                              print_reports: bool = True,     # <-- nuevo
                              save_path: str | None = None):
    """
    Pinta matrices de confusión en grid de 2 columnas (máx. 8) y
    opcionalmente imprime el classification_report por símbolo.
    Etiquetas DOWN/UP. Color de texto adaptativo (amarillo↔violeta).
    - normalize=True -> proporciones por clase (recall por fila)
    - per_axis_scale=True -> cada subplot con su propia escala de colores y su colorbar
    """
    symbols = symbols[:8]
    df = pd.read_csv(csv_path, sep=";")

    # Prepara matrices y pares (y_true, y_pred) para los reports
    cms_show, pairs = [], []
    for sym in symbols:
        d = df[df["Symbol"] == sym]
        if d.empty:
            cms_show.append(None); pairs.append((None, None)); continue
        y_true, y_pred = _reconstruct_ytrue(d)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).astype(float)
        if normalize:
            row_sums = cm.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            cm_show = cm / row_sums
        else:
            cm_show = cm
        cms_show.append(cm_show)
        pairs.append((y_true, y_pred))

    cols = 2
    rows = (len(symbols) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7.2*cols, 5.2*rows))
    if rows == 1:
        axes = np.array([axes])
    cmap = plt.get_cmap("viridis")

    # Escala global si se pide
    global_vmax = None
    if not per_axis_scale:
        global_vmax = np.nanmax([cm.max() for cm in cms_show if cm is not None])

    for idx, sym in enumerate(symbols):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        cm_show = cms_show[idx]
        y_true, y_pred = pairs[idx]

        if cm_show is None or y_true is None:
            ax.axis("off"); continue

        # vmin/vmax por eje o global
        vmin, vmax = (None, None) if per_axis_scale else (0.0, global_vmax)
        im = ax.imshow(cm_show, cmap=cmap, vmin=vmin, vmax=vmax)

        ax.set_title(f"{sym}", pad=8)
        ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["DOWN","UP"]); ax.set_yticklabels(["DOWN","UP"])

        # Texto adaptativo + contorno
        norm = im.norm
        for i in range(2):
            for j in range(2):
                val = cm_show[i, j]
                txt_color = "#5B2C6F" if norm(val) >= 0.6 else "#FFD700"
                t = ax.text(j, i, format(val, ".2f" if normalize else ".0f"),
                            ha="center", va="center",
                            color=txt_color, fontsize=12, fontweight="bold")
                t.set_path_effects([
                    patheffects.Stroke(linewidth=1.8, foreground="black", alpha=0.35),
                    patheffects.Normal()
                ])

        # cuadrícula sutil
        ax.set_xticks(np.arange(-.5, 2, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 2, 1), minor=True)
        ax.grid(which="minor", color="w", linestyle="-", linewidth=0.5, alpha=0.3)
        ax.tick_params(which="minor", bottom=False, left=False)

        # Colorbar por subplot si per_axis_scale=True
        if per_axis_scale:
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label("Proportion" if normalize else "Count", rotation=90, va="center")

        # === Classification report (consola) ===
        if print_reports:
            print(f"\n=== Classification report for {sym} ===")
            print(classification_report(y_true, y_pred, digits=3, zero_division=0))

    # Apaga huecos
    total_axes = rows*cols
    for k in range(len(symbols), total_axes):
        r, c = divmod(k, cols)
        axes[r, c].axis("off")

    fig.tight_layout()

    # Colorbar común si usamos escala global
    if not per_axis_scale:
        cb = fig.colorbar(axes[0,0].images[0], ax=axes, shrink=0.9, location="right", pad=0.02)
        cb.set_label("Proportion" if normalize else "Count", rotation=90, va="center")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[OK] Guardado: {save_path}")

    plt.show()

if __name__ == "__main__":
    symbols = ["NVDA","MSFT","MU","AAPL"]  # máx. 8
    plot_confusion_grid_2cols(
        CSV_PATH,
        symbols=symbols,
        normalize=False,        # True → proporciones por clase
        per_axis_scale=True,    # cada subplot con su propia escala y colorbar
        print_reports=True,     # imprime un report por símbolo
        save_path=None          # p.ej. "Resultados/CM_GRID_2cols_peraxis.png"
    )
