import numpy as np


def binary_cross_entropy(y_true, y_pred):
    """
    Binary Cross-Entropy loss.

    Parameters
    ----------
    y_true : array-like
        True binary labels.

    y_pred : array-like
        Predicted probabilities.

    Returns
    -------
    float
        Mean binary cross-entropy.
    """

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    epsilon = 1e-15

    y_pred = np.clip(
        y_pred,
        epsilon,
        1.0 - epsilon
    )

    loss = (
        y_true * np.log(y_pred)
        +
        (1.0 - y_true) * np.log(1.0 - y_pred)
    )

    return -np.mean(loss)
def mean_squared_error(y_true, y_pred):
    """
    Compute Mean Squared Error.

    MSE = mean((y_true - y_pred)^2)
    """

    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    return np.mean(
        (y_true - y_pred) ** 2
    )