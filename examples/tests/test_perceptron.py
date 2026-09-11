import numpy as np

from src.engine.perceptron import Perceptron


# ==========================
# TEST XOR
# ==========================

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([
    0,
    1,
    1,
    0
])


model = Perceptron(
    learning_rate=0.1,
    epochs=20
)

model.fit(X, y)

print("\n========== XOR RESULTS ==========")

print("Weights      :", model.weights)
print("Bias         :", model.bias)
print("Predictions  :", model.predict(X))
print("True labels  :", y)
print("Accuracy     :", model.score(X, y))
print("Errors/epoch :", model.errors_)