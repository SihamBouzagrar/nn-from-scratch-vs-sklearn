import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.engine.mlp_classifier import MLPClassifierScratch


# =====================================
# 1. Generate dataset
# =====================================

X, y = make_moons(
    n_samples=1000,
    noise=0.2,
    random_state=42
)

print("X shape:", X.shape)
print("y shape:", y.shape)


# =====================================
# 2. Train / Test split
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =====================================
# 3. Standardization
# =====================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


# =====================================
# 4. Create model
# =====================================

model = MLPClassifierScratch(
    hidden_layer_size=8,
    learning_rate=0.05,
    epochs=5000
)


# =====================================
# 5. Train
# =====================================

model.fit(
    X_train,
    y_train
)


# =====================================
# 6. Predictions
# =====================================

y_pred = model.predict(X_test)


# =====================================
# 7. Accuracy
# =====================================

accuracy = np.mean(
    y_pred == y_test
)


print("\n========== RESULTS ==========")

print("Accuracy:", accuracy)


# =====================================
# 8. Loss curve
# =====================================

plt.figure()

plt.plot(
    model.loss_history_
)

plt.xlabel("Epoch")

plt.ylabel("Binary Cross-Entropy")

plt.title(
    "MLP Scratch - Training Loss"
)

plt.grid()

plt.show()