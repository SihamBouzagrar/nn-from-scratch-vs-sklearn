import numpy as np

from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

from src.engine.mlp_classifier import (
    MLPClassifierScratch
)

from src.pipeline.kfold import KFoldScratch


# =====================================
# Dataset
# =====================================

X, y = make_moons(
    n_samples=500,
    noise=0.2,
    random_state=42
)


# =====================================
# Standardization
# =====================================

scaler = StandardScaler()

X = scaler.fit_transform(X)


# =====================================
# K-Fold
# =====================================

cv = KFoldScratch(
    n_splits=5,
    shuffle=True,
    random_state=42
)


scores = []


# =====================================
# Cross-validation
# =====================================

for fold, (
    train_idx,
    test_idx
) in enumerate(
    cv.split(X),
    start=1
):

    X_train = X[train_idx]
    X_test = X[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]


    # New model for each fold
    model = MLPClassifierScratch(
        hidden_layer_size=8,
        learning_rate=0.05,
        epochs=3000
    )


    model.fit(
        X_train,
        y_train
    )


    predictions = model.predict(
        X_test
    )


    accuracy = np.mean(
        predictions == y_test
    )


    scores.append(accuracy)


    print(
        f"Fold {fold}: "
        f"accuracy = {accuracy:.4f}"
    )


# =====================================
# Final results
# =====================================

scores = np.array(scores)

print("\n========== CROSS-VALIDATION ==========")

print("Scores:", scores)

print(
    f"Mean accuracy: "
    f"{scores.mean():.4f}"
)

print(
    f"Std accuracy: "
    f"{scores.std():.4f}"
)