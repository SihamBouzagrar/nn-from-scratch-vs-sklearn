import numpy as np


def sigmoid(x):
    """
    Sigmoid activation function.

    sigma(x) = 1 / (1 + exp(-x))
    """
    x = np.asarray(x)

    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x):
    """
    Derivative of sigmoid.
    """
    s = sigmoid(x)

    return s * (1.0 - s)


def tanh(x):
    """
    Hyperbolic tangent activation function.
    """
    return np.tanh(x)


def tanh_derivative(x):
    """
    Derivative of tanh.
    """
    t = np.tanh(x)

    return 1.0 - t**2


def relu(x):
    """
    ReLU activation function.
    """
    x = np.asarray(x)

    return np.maximum(0, x)


def relu_derivative(x):
    """
    Derivative of ReLU.
    """
    x = np.asarray(x)

    return (x > 0).astype(float)


def identity(x):
    """
    Identity activation function.
    """
    return np.asarray(x)


def identity_derivative(x):
    """
    Derivative of identity.
    """
    x = np.asarray(x)

    return np.ones_like(x, dtype=float)