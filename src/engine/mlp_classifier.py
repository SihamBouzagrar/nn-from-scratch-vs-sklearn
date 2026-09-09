import numpy as np

from src.engine.activations import (
    relu,
    sigmoid,
)


class MLPClassifierScratch:
    """
    Simple Multi-Layer Perceptron classifier implemented from scratch.

    Architecture:
        input -> hidden -> output
    """

    def __init__(
        self,
        hidden_layer_size=2,
        learning_rate=0.1,
        epochs=1000,
    ):
        self.hidden_layer_size = hidden_layer_size
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.weights_input_hidden = None
        self.bias_hidden = None

        self.weights_hidden_output = None
        self.bias_output = None

    def _initialize_parameters(self, n_features):
        """
        Initialize weights and biases.
        """

        self.weights_input_hidden = (
            np.random.randn(
                n_features,
                self.hidden_layer_size
            ) * 0.1
        )

        self.bias_hidden = np.zeros(
            self.hidden_layer_size
        )

        self.weights_hidden_output = (
            np.random.randn(
                self.hidden_layer_size,
                1
            ) * 0.1
        )

        self.bias_output = np.zeros(1)

    def _forward(self, X):
        """
        Forward propagation.
        """

        # Hidden layer
        z_hidden = (
            X @ self.weights_input_hidden
            + self.bias_hidden
        )

        a_hidden = relu(z_hidden)

        # Output layer
        z_output = (
            a_hidden @ self.weights_hidden_output
            + self.bias_output
        )

        a_output = sigmoid(z_output)

        return (
            z_hidden,
            a_hidden,
            z_output,
            a_output
        )

    def predict_proba(self, X):
        """
        Return output probabilities.
        """

        X = np.asarray(X)

        if self.weights_input_hidden is None:
            self._initialize_parameters(
                X.shape[1]
            )

        _, _, _, a_output = self._forward(X)

        return a_output.ravel()

    def predict(self, X):
        """
        Convert probabilities into binary predictions.
        """

        probabilities = self.predict_proba(X)

        return (
            probabilities >= 0.5
        ).astype(int)