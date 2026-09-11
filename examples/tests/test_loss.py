import numpy as np

from src.engine.losses import binary_cross_entropy


y_true = np.array([1, 0, 1, 1])

y_pred_good = np.array([
    0.9,
    0.1,
    0.8,
    0.95
])

y_pred_bad = np.array([
    0.1,
    0.9,
    0.2,
    0.05
])


print("========== LOSS TEST ==========")

loss_good = binary_cross_entropy(
    y_true,
    y_pred_good
)

loss_bad = binary_cross_entropy(
    y_true,
    y_pred_bad
)

print("Good predictions:")
print("Loss =", loss_good)

print("\nBad predictions:")
print("Loss =", loss_bad)