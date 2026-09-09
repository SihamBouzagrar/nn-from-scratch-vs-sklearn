import numpy as np

from src.engine.mlp_classifier import MLPClassifierScratch
from src.engine.losses import binary_cross_entropy


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


model = MLPClassifierScratch(
    hidden_layer_size=2
)

model._initialize_parameters(
    n_features=2
)


# ==========================
# FORWARD PASS
# ==========================

z_hidden, a_hidden, z_output, a_output = (
    model._forward(X)
)

# a_output = predictions
y_pred = a_output


# ==========================
# LOSS
# ==========================

loss = binary_cross_entropy(
    y,
    y_pred.ravel()
)


print("\n========== MLP FORWARD PASS ==========")

print("\nX:")
print(X)
print("Shape:", X.shape)

print("\nW1:")
print(model.weights_input_hidden)
print("Shape:", model.weights_input_hidden.shape)

print("\nb1:")
print(model.bias_hidden)
print("Shape:", model.bias_hidden.shape)

print("\nZ_hidden:")
print(z_hidden)
print("Shape:", z_hidden.shape)

print("\nA_hidden:")
print(a_hidden)
print("Shape:", a_hidden.shape)

print("\nW2:")
print(model.weights_hidden_output)
print("Shape:", model.weights_hidden_output.shape)

print("\nb2:")
print(model.bias_output)
print("Shape:", model.bias_output.shape)

print("\nZ_output:")
print(z_output)
print("Shape:", z_output.shape)

print("\nA_output / Predictions:")
print(a_output)
print("Shape:", a_output.shape)

print("\nLoss:")
print(loss)