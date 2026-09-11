import time
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
)
from sklearn.pipeline import Pipeline

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


# ============================================================
# 1. Dataset
# ============================================================

X, y = make_moons(
    n_samples=500,
    noise=0.2,
    random_state=42
)


# ============================================================
# 2. Hold-out TEST set
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================================================
# 3. Hyperparameter grid
# ============================================================

param_grid = {
    "learning_rate": [
        0.01,
        0.05
    ],

    "hidden_layer_size": [
        4,
        8
    ],

    "epochs": [
        1000,
        3000
    ]
}


# ============================================================
# 4. OUR GRID SEARCH
# ============================================================

scratch_cv = StratifiedKFoldScratch(
    n_splits=5,
    shuffle=True,
    random_state=42
)


scratch_grid = GridSearchScratch(
    model_class=MLPClassifierScratch,
    param_grid=param_grid,
    cv=scratch_cv,
    scaler_class=StandardScaler,
)


start = time.perf_counter()

scratch_grid.fit(
    X_train,
    y_train
)

scratch_time = (
    time.perf_counter() - start
)


scratch_test_predictions = (
    scratch_grid.predict(X_test)
)

scratch_test_accuracy = (
    (scratch_test_predictions == y_test)
    .mean()
)

# ============================================================
# SCIKIT-LEARN GRID SEARCH
# ============================================================

sklearn_pipeline = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "mlp",
        MLPClassifier(
            random_state=42,
            max_iter=3000
        )
    )
])


sklearn_param_grid = {
    "mlp__learning_rate_init": [
        0.01,
        0.05
    ],

    "mlp__hidden_layer_sizes": [
        (4,),
        (8,)
    ],

    "mlp__max_iter": [
        1000,
        3000
    ]
}


sklearn_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


sklearn_grid = GridSearchCV(
    estimator=sklearn_pipeline,
    param_grid=sklearn_param_grid,
    cv=sklearn_cv,
    scoring="accuracy",
    n_jobs=1,
    return_train_score=True
)


start = time.perf_counter()

sklearn_grid.fit(
    X_train,
    y_train
)

sklearn_time = (
    time.perf_counter() - start
)


sklearn_test_accuracy = (
    sklearn_grid.score(
        X_test,
        y_test
    )
)


# ============================================================
# 6. RESULTS
# ============================================================

print("\n========================================")
print("GRID SEARCH COMPARISON")
print("========================================")

print("\nScratch implementation")
print("----------------------------------------")

print(
    "Best params:",
    scratch_grid.best_params_
)

print(
    "Best CV score:",
    scratch_grid.best_score_
)

print(
    "Test accuracy:",
    scratch_test_accuracy
)

print(
    "Execution time:",
    scratch_time
)


print("\nScikit-Learn GridSearchCV")
print("----------------------------------------")

print(
    "Best params:",
    sklearn_grid.best_params_
)

print(
    "Best CV score:",
    sklearn_grid.best_score_
)

print(
    "Test accuracy:",
    sklearn_test_accuracy
)

print(
    "Execution time:",
    sklearn_time
)