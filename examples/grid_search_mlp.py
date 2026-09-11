from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

from src.engine.mlp_classifier import (
    MLPClassifierScratch
)

from src.pipeline.stratified_kfold import (
    StratifiedKFoldScratch
)

from src.pipeline.grid_search import (
    GridSearchScratch
)


# =========================================
# Dataset
# =========================================

X, y = make_moons(
    n_samples=500,
    noise=0.2,
    random_state=42
)


# =========================================
# CV
# =========================================

cv = StratifiedKFoldScratch(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# =========================================
# Grid
# =========================================

param_grid = {
    "learning_rate": [
        0.001,
        0.01,
        0.05
    ],

    "hidden_layer_size": [
        4,
        8,
        16
    ],

    "epochs": [
        1000,
        3000
    ]
}


# =========================================
# Grid Search
# =========================================

grid = GridSearchScratch(
    model_class=MLPClassifierScratch,
    param_grid=param_grid,
    cv=cv,
    scaler_class=StandardScaler,
)


grid.fit(X, y)


# =========================================
# Results
# =========================================

print("\n========== GRID SEARCH RESULTS ==========")

print(
    "Best parameters:"
)

print(
    grid.best_params_
)

print(
    "\nBest CV score:"
)

print(
    grid.best_score_
)