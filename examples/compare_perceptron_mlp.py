import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from src.engine.perceptron import Perceptron
from src.engine.mlp_classifier import MLPClassifierScratch


# ============================================================
# 1. Generate dataset
# ============================================================

X, y = make_moons(
    n_samples=1000,
    noise=0.2,
    random_state=42
)


# ============================================================
# 2. Train / Test split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================================================
# 3. Standardization
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# 4. Perceptron
# ============================================================

perceptron = Perceptron(
    learning_rate=0.01,
    epochs=100
)

perceptron.fit(
    X_train_scaled,
    y_train
)

y_pred_perceptron = perceptron.predict(
    X_test_scaled
)

accuracy_perceptron = accuracy_score(
    y_test,
    y_pred_perceptron
)


# ============================================================
# 5. MLP
# ============================================================

mlp = MLPClassifierScratch(
    hidden_layer_size=8,
    learning_rate=0.05,
    epochs=5000
)

mlp.fit(
    X_train_scaled,
    y_train
)

y_pred_mlp = mlp.predict(
    X_test_scaled
)

accuracy_mlp = accuracy_score(
    y_test,
    y_pred_mlp
)


# ============================================================
# 6. Print results
# ============================================================

print("\n========== MODEL COMPARISON ==========")

print(
    f"Perceptron accuracy : "
    f"{accuracy_perceptron:.4f}"
)

print(
    f"MLP accuracy        : "
    f"{accuracy_mlp:.4f}"
)


# ============================================================
# 7. Create decision grid
# ============================================================

x_min = X[:, 0].min() - 0.5
x_max = X[:, 0].max() + 0.5

y_min = X[:, 1].min() - 0.5
y_max = X[:, 1].max() + 0.5

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

grid = np.c_[
    xx.ravel(),
    yy.ravel()
]

grid_scaled = scaler.transform(grid)


# ============================================================
# 8. Perceptron decision boundary
# ============================================================

Z_perceptron = perceptron.predict(
    grid_scaled
)

Z_perceptron = Z_perceptron.reshape(
    xx.shape
)


plt.figure(figsize=(8, 6))

plt.contourf(
    xx,
    yy,
    Z_perceptron,
    alpha=0.25
)

plt.scatter(
    X_test[:, 0],
    X_test[:, 1],
    c=y_test
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.title(
    "Perceptron - Decision Boundary"
)

plt.show()


# ============================================================
# 9. MLP decision boundary
# ============================================================

Z_mlp = mlp.predict(
    grid_scaled
)

Z_mlp = Z_mlp.reshape(
    xx.shape
)


plt.figure(figsize=(8, 6))

plt.contourf(
    xx,
    yy,
    Z_mlp,
    alpha=0.25
)

plt.scatter(
    X_test[:, 0],
    X_test[:, 1],
    c=y_test
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.title(
    "MLP Scratch - Decision Boundary"
)

plt.show()