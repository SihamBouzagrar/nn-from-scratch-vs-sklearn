import numpy as np
import matplotlib.pyplot as plt

from src.engine.mlp_classifier import MLPClassifierScratch


np.random.seed(42)


X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=float)

y = np.array([
    0,
    1,
    1,
    0
])


model = MLPClassifierScratch(
    hidden_layer_size=4,
    #2==>4===>1
    learning_rate=0.1,
    epochs=10000
)


model.fit(X, y)


print("\n========== XOR ==========")

print("Probabilities:")
print(model.predict_proba(X))

print("\nPredictions:")
print(model.predict(X))

print("\nTrue labels:")
print(y)

print("\nAccuracy:")
print(np.mean(model.predict(X) == y))


plt.plot(model.loss_history_)
plt.xlabel("Epoch")
plt.ylabel("Binary Cross-Entropy")
plt.title("MLP Training Loss on XOR")
plt.grid()
plt.show()