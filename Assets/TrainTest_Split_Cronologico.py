import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Datos
years = [str(y) for y in range(2010, 2025)]
n_folds = 5
test_years = [str(y) for y in range(2020, 2025)]
colors_train = "#aed9e0"  # Azul claro
colors_test = "#ffb677"   # Naranja pastel

fig, axes = plt.subplots(n_folds, 1, figsize=(12, 2*n_folds), sharex=True)

for i, ax in enumerate(axes):
    fold_train_years = years[:10+i]  # Los años de train van creciendo
    fold_test_year = test_years[i]
    for j, year in enumerate(years):
        if year in fold_train_years:
            color = colors_train
        elif year == fold_test_year:
            color = colors_test
        else:
            color = "#f7f7f7"  # Gris clarito para el futuro (opcional)
        rect = plt.Rectangle((j, 0), 1, 1, facecolor=color, edgecolor='k')
        ax.add_patch(rect)
        ax.text(j+0.5, 0.5, year, ha='center', va='center', fontsize=11)
    # Etiquetas de fold
    ax.text(-1.5, 0.5, f"Fold {i+1}", va='center', ha='right', fontsize=12, fontweight='bold')
    ax.set_xlim(-2, len(years))
    ax.set_ylim(0, 1)
    ax.axis('off')

# Leyenda
train_patch = mpatches.Patch(color=colors_train, label='Train')
test_patch = mpatches.Patch(color=colors_test, label='Test')
plt.legend(handles=[train_patch, test_patch], bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=12)

plt.suptitle("Validación cruzada cronológica en 5 folds", fontsize=15, y=1.02)
plt.tight_layout()
plt.show()
