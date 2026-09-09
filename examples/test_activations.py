import numpy as np

from src.engine.activations import (
    sigmoid,
    sigmoid_derivative,
    tanh,
    tanh_derivative,
    relu,
    relu_derivative,
    identity,
    identity_derivative,
)


x = np.array([
    -2.0,
    -1.0,
    0.0,
    1.0,
    2.0
])


print("x:")
print(x)

print("\n========== SIGMOID ==========")
print("values     :", sigmoid(x))
print("derivative :", sigmoid_derivative(x))

print("\n========== TANH ==========")
print("values     :", tanh(x))
print("derivative :", tanh_derivative(x))

print("\n========== RELU ==========")
print("values     :", relu(x))
print("derivative :", relu_derivative(x))

print("\n========== IDENTITY ==========")
print("values     :", identity(x))
print("derivative :", identity_derivative(x))