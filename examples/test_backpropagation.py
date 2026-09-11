import numpy as np

from src.engine.mlp_classifier import MLPClassifierScratch
from src.engine.losses import binary_cross_entropy


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
], dtype=float)


model = MLPClassifierScratch(
    hidden_layer_size=2,
    learning_rate=0.1,
    epochs=1
)


model._initialize_parameters(
    n_features=2
)


# Forward
z_hidden, a_hidden, z_output, a_output = (
    model._forward(X)
)


loss = binary_cross_entropy(
    y,
    a_output.ravel()
)


# Backward
(
    dW_hidden,
    db_hidden,
    dW_output,
    db_output,
) = model._backward(
    X,
    y,
    z_hidden,
    a_hidden,
    a_output,
)


print("\n========== BACKPROPAGATION ==========")

print("\nLoss:")
print(loss)

print("\nW1:")
print(model.weights_input_hidden)

print("\ndW1:")
print(dW_hidden)

print("\nShape dW1:")
print(dW_hidden.shape)

print("\nW2:")
print(model.weights_hidden_output)

print("\ndW2:")
print(dW_output)

print("\nShape dW2:")
print(dW_output.shape)

print("\ndb1:")
print(db_hidden)

print("\ndb2:")
print(db_output)