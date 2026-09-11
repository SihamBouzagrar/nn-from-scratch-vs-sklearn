from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

from src.engine.mlp_classifier import (
    MLPClassifierScratch
)

from src.pipeline.stratified_kfold import (
    StratifiedKFoldScratch
)

from src.pipeline.cross_validation import (
    cross_validate
)


X, y = make_moons(
    n_samples=500,
    noise=0.2,
    random_state=42
)


cv = StratifiedKFoldScratch(
    n_splits=5,
    shuffle=True,
    random_state=42
)


results = cross_validate(
    model_class=MLPClassifierScratch,
    X=X,
    y=y,
    cv=cv,
    model_params={
        "hidden_layer_size": 8,
        "learning_rate": 0.05,
        "epochs": 3000,
    },
    scaler_class=StandardScaler,
)


print("\n========== GENERIC CV ==========")

print(
    "Scores:",
    results["scores"]
)

print(
    f"Mean:",
    f"{results['mean']:.4f}"
)

print(
    f"Std:",
    f"{results['std']:.4f}"
)