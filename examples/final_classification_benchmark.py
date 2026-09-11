import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from src.engine.mlp_classifier import MLPClassifierScratch


# ============================================================
# 1. DATASET
# ============================================================

X, y = make_moons(
    n_samples=1000,
    noise=0.2,
    random_state=42
)


# ============================================================
# 2. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================================================
# 3. STANDARDIZATION
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# 4. SCRATCH MODEL
# ============================================================

scratch_model = MLPClassifierScratch(
    hidden_layer_size=8,
    learning_rate=0.01,
    epochs=1000,
    random_state=42,
)


print("\nTraining Scratch MLP...")

start = time.perf_counter()

scratch_model.fit(
    X_train_scaled,
    y_train
)

scratch_time = (
    time.perf_counter() - start
)


scratch_pred = scratch_model.predict(
    X_test_scaled
)


# ============================================================
# 5. SCIKIT-LEARN MODEL
# ============================================================

sklearn_model = MLPClassifier(
    hidden_layer_sizes=(8,),
    activation="relu",
    solver="adam",
    learning_rate_init=0.01,
    max_iter=1000,
    random_state=42,
)


print("Training Scikit-Learn MLP...")

start = time.perf_counter()

sklearn_model.fit(
    X_train_scaled,
    y_train
)

sklearn_time = (
    time.perf_counter() - start
)


sklearn_pred = sklearn_model.predict(
    X_test_scaled
)


# ============================================================
# 6. METRICS
# ============================================================

scratch_results = {
    "accuracy": accuracy_score(
        y_test,
        scratch_pred
    ),
    "precision": precision_score(
        y_test,
        scratch_pred,
        zero_division=0
    ),
    "recall": recall_score(
        y_test,
        scratch_pred,
        zero_division=0
    ),
    "f1": f1_score(
        y_test,
        scratch_pred,
        zero_division=0
    ),
}


sklearn_results = {
    "accuracy": accuracy_score(
        y_test,
        sklearn_pred
    ),
    "precision": precision_score(
        y_test,
        sklearn_pred,
        zero_division=0
    ),
    "recall": recall_score(
        y_test,
        sklearn_pred,
        zero_division=0
    ),
    "f1": f1_score(
        y_test,
        sklearn_pred,
        zero_division=0
    ),
}


# ============================================================
# 7. DISPLAY RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("FINAL CLASSIFICATION BENCHMARK")
print("=" * 60)


print("\nScratch MLP")
print("-" * 60)

print(
    f"Accuracy     : "
    f"{scratch_results['accuracy']:.4f}"
)

print(
    f"Precision    : "
    f"{scratch_results['precision']:.4f}"
)

print(
    f"Recall       : "
    f"{scratch_results['recall']:.4f}"
)

print(
    f"F1-score     : "
    f"{scratch_results['f1']:.4f}"
)

print(
    f"Training time: "
    f"{scratch_time:.6f} s"
)

print(
    f"Epochs used  : "
    f"{len(scratch_model.loss_history_)}"
)


print("\nScikit-Learn MLP")
print("-" * 60)

print(
    f"Accuracy     : "
    f"{sklearn_results['accuracy']:.4f}"
)

print(
    f"Precision    : "
    f"{sklearn_results['precision']:.4f}"
)

print(
    f"Recall       : "
    f"{sklearn_results['recall']:.4f}"
)

print(
    f"F1-score     : "
    f"{sklearn_results['f1']:.4f}"
)

print(
    f"Training time: "
    f"{sklearn_time:.6f} s"
)

print(
    f"Iterations   : "
    f"{sklearn_model.n_iter_}"
)


# ============================================================
# 8. COMPARISON
# ============================================================

print("\n")
print("=" * 60)
print("COMPARISON")
print("=" * 60)

print(
    f"Accuracy difference : "
    f"{abs(scratch_results['accuracy'] - sklearn_results['accuracy']):.4f}"
)

print(
    f"Training time ratio : "
    f"{scratch_time / sklearn_time:.2f}x"
)


# ============================================================
# 9. LOSS CURVES
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    scratch_model.loss_history_,
    label="Scratch MLP"
)

plt.plot(
    sklearn_model.loss_curve_,
    label="Scikit-Learn MLP"
)

plt.xlabel("Iteration / Epoch")
plt.ylabel("Loss")
plt.title(
    "Training Loss — Scratch vs Scikit-Learn"
)

plt.legend()
plt.grid()

plt.show()