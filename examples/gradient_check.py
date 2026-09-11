import numpy as np

from src.engine.mlp_classifier import (
    MLPClassifierScratch,
)

from src.engine.losses import (
    binary_cross_entropy,
)


np.random.seed(42)


X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
], dtype=float)

y = np.array([
    0,
    1,
    1,
    0,
], dtype=float)


model = MLPClassifierScratch(
    hidden_layer_size=3,
    learning_rate=0.1,
    epochs=1,
)

model._initialize_parameters(
    n_features=2
)


# =====================================
# Analytical gradients
# =====================================

z_hidden, a_hidden, z_output, a_output = (
    model._forward(X)
)

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


# =====================================
# Numerical gradient for W1[0, 0]
# =====================================

epsilon = 1e-5


original_value = (
    model.weights_input_hidden[0, 0]
)


# J(w + epsilon)
model.weights_input_hidden[0, 0] = (
    original_value + epsilon
)

_, _, _, output_plus = model._forward(X)

loss_plus = binary_cross_entropy(
    y,
    output_plus.ravel()
)


# J(w - epsilon)
model.weights_input_hidden[0, 0] = (
    original_value - epsilon
)

_, _, _, output_minus = model._forward(X)

loss_minus = binary_cross_entropy(
    y,
    output_minus.ravel()
)


# Restore original value
model.weights_input_hidden[0, 0] = original_value


numerical_gradient = (
    loss_plus - loss_minus
) / (2 * epsilon)


analytical_gradient = dW_hidden[0, 0]


# =====================================
# Comparison
# =====================================

print("\n========== GRADIENT CHECK ==========")

print("\nAnalytical gradient:")
print(analytical_gradient)

print("\nNumerical gradient:")
print(numerical_gradient)

difference = abs(
    analytical_gradient
    - numerical_gradient
)

print("\nAbsolute difference:")
print(difference)