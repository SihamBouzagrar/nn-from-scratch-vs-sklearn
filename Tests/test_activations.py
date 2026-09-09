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


def test_sigmoid_zero():
    assert np.isclose(sigmoid(0), 0.5)


def test_sigmoid_range():
    x = np.array([-2, -1, 0, 1, 2])
    result = sigmoid(x)

    assert np.all(result >= 0)
    assert np.all(result <= 1)


def test_relu():
    x = np.array([-2, -1, 0, 1, 2])

    expected = np.array([0, 0, 0, 1, 2])

    assert np.array_equal(relu(x), expected)


def test_identity():
    x = np.array([-2, -1, 0, 1, 2])

    assert np.array_equal(identity(x), x)


def test_relu_derivative():
    x = np.array([-2, -1, 0, 1, 2])

    expected = np.array([0, 0, 0, 1, 1])

    assert np.array_equal(
        relu_derivative(x),
        expected
    )