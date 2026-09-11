
import numpy as np


from src.engine.activations import (
    relu,
    relu_derivative,
    identity,
)
from src.engine.losses import mean_squared_error


class MLPRegressorScratch:
    """
    Simple Multi-Layer Perceptron regressor implemented from scratch.

    Architecture:
        input -> hidden -> output

    Hidden layer:
        ReLU activation

    Output layer:
        Identity activation

    Loss:
        Mean Squared Error (MSE)

    Optimization:
        Batch Gradient Descent
    """

    def __init__(
        self,
        hidden_layer_size=10,
        learning_rate=0.01,
        epochs=1000,
        random_state=None,
    ):
        """
        Initialize the MLP regressor.

        Parameters
        ----------
        hidden_layer_size : int
            Number of neurons in the hidden layer.

        learning_rate : float
            Learning rate used by gradient descent.

        epochs : int
            Number of training iterations.

        random_state : int or None
            Seed used for reproducible weight initialization.
        """

        self.hidden_layer_size = hidden_layer_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_state = random_state

        # Model parameters
        self.weights_input_hidden = None
        self.bias_hidden = None

        self.weights_hidden_output = None
        self.bias_output = None

        # Training history
        self.loss_history_ = []

    def _initialize_parameters(self, n_features):
        """
        Initialize weights and biases.
        """

        if self.random_state is not None:
            rng = np.random.default_rng(
                self.random_state
            )
        else:
            rng = np.random.default_rng()

        # =====================================
        # Input -> Hidden
        # =====================================

        self.weights_input_hidden = (
            rng.standard_normal(
                (
                    n_features,
                    self.hidden_layer_size,
                )
            )
            * 0.1
        )

        self.bias_hidden = np.zeros(
            self.hidden_layer_size
        )

        # =====================================
        # Hidden -> Output
        # =====================================

        self.weights_hidden_output = (
            rng.standard_normal(
                (
                    self.hidden_layer_size,
                    1,
                )
            )
            * 0.1
        )

        self.bias_output = np.zeros(1)

    def _forward(self, X):
        """
        Forward propagation.

        Parameters
        ----------
        X : ndarray
            Input features.

        Returns
        -------
        z_hidden : ndarray
            Weighted sum of hidden layer.

        a_hidden : ndarray
            ReLU activation of hidden layer.

        z_output : ndarray
            Weighted sum of output layer.

        a_output : ndarray
            Final regression prediction.
        """

        # =====================================
        # 1. Hidden layer
        # =====================================

        z_hidden = (
            X @ self.weights_input_hidden
            + self.bias_hidden
        )

        a_hidden = relu(z_hidden)

        # =====================================
        # 2. Output layer
        # =====================================

        z_output = (
            a_hidden @ self.weights_hidden_output
            + self.bias_output
        )

        # Identity activation
        a_output = identity(z_output)

        return (
            z_hidden,
            a_hidden,
            z_output,
            a_output,
        )

    def _backward(
        self,
        X,
        y,
        z_hidden,
        a_hidden,
        a_output,
    ):
        """
        Backpropagation.

        Computes the gradients of the MSE loss
        with respect to weights and biases.

        For MSE:

            L = 1/n * sum((y_pred - y)^2)

        Therefore:

            dL/dA_output = 2/n * (y_pred - y)

        Since the output activation is Identity:

            dA_output/dZ_output = 1

        Therefore:

            dL/dZ_output = dL/dA_output
        """

        X = np.asarray(
            X,
            dtype=float,
        )

        y = np.asarray(
            y,
            dtype=float,
        ).reshape(-1, 1)

        n_samples = X.shape[0]

        # =====================================
        # 1. Output layer error
        # =====================================

        # dL/dZ_output
        delta_output = (
            2.0
            * (a_output - y)
            / n_samples
        )

        # =====================================
        # 2. Output layer gradients
        # =====================================

        # dL/dW2
        dW_output = (
            a_hidden.T @ delta_output
        )

        # dL/db2
        db_output = np.sum(
            delta_output,
            axis=0,
        )

        # =====================================
        # 3. Propagate to hidden layer
        # =====================================

        # dL/dA_hidden
        dA_hidden = (
            delta_output
            @ self.weights_hidden_output.T
        )

        # =====================================
        # 4. Backpropagation through ReLU
        # =====================================

        # dL/dZ_hidden
        delta_hidden = (
            dA_hidden
            * relu_derivative(z_hidden)
        )

        # =====================================
        # 5. Hidden layer gradients
        # =====================================

        # dL/dW1
        dW_hidden = (
            X.T @ delta_hidden
        )

        # dL/db1
        db_hidden = np.sum(
            delta_hidden,
            axis=0,
        )

        return (
            dW_hidden,
            db_hidden,
            dW_output,
            db_output,
        )

    def _update_parameters(
        self,
        dW_hidden,
        db_hidden,
        dW_output,
        db_output,
    ):
        """
        Update weights and biases using
        gradient descent.
        """

        # =====================================
        # Hidden layer
        # =====================================

        self.weights_input_hidden -= (
            self.learning_rate
            * dW_hidden
        )

        self.bias_hidden -= (
            self.learning_rate
            * db_hidden
        )

        # =====================================
        # Output layer
        # =====================================

        self.weights_hidden_output -= (
            self.learning_rate
            * dW_output
        )

        self.bias_output -= (
            self.learning_rate
            * db_output
        )

    def _compute_loss(self, y, a_output):
        """
        Compute Mean Squared Error.
        """

        return mean_squared_error(
            y,
            a_output.ravel(),
        )

    def fit(self, X, y):
        """
        Train the MLP regressor using
        batch gradient descent.

        Parameters
        ----------
        X : ndarray
            Training features.

        y : ndarray
            Target values.

        Returns
        -------
        self
            Fitted model.
        """

        X = np.asarray(
            X,
            dtype=float,
        )

        y = np.asarray(
            y,
            dtype=float,
        )

        n_samples, n_features = X.shape

        # =====================================
        # 1. Initialize parameters
        # =====================================

        self._initialize_parameters(
            n_features
        )

        # Reset loss history
        self.loss_history_ = []

        # =====================================
        # 2. Training loop
        # =====================================

        for epoch in range(self.epochs):

            # ---------------------------------
            # FORWARD
            # ---------------------------------

            (
                z_hidden,
                a_hidden,
                z_output,
                a_output,
            ) = self._forward(X)

            # ---------------------------------
            # LOSS
            # ---------------------------------

            loss = self._compute_loss(
                y,
                a_output,
            )

            self.loss_history_.append(
                loss
            )

            # ---------------------------------
            # BACKWARD
            # ---------------------------------

            (
                dW_hidden,
                db_hidden,
                dW_output,
                db_output,
            ) = self._backward(
                X,
                y,
                z_hidden,
                a_hidden,
                a_output,
            )

            # ---------------------------------
            # UPDATE
            # ---------------------------------

            self._update_parameters(
                dW_hidden,
                db_hidden,
                dW_output,
                db_output,
            )

        return self

    def predict(self, X):
        """
        Return continuous predictions.
        """

        X = np.asarray(
            X,
            dtype=float,
        )

        if self.weights_input_hidden is None:
            raise ValueError(
                "The model must be fitted before prediction."
            )

        _, _, _, a_output = self._forward(X)

        return a_output.ravel()

    def score(self, X, y):
        """
        Return the coefficient of determination R².

        R² = 1 - SS_res / SS_tot
        """

        y = np.asarray(
            y,
            dtype=float,
        )

        predictions = self.predict(X)

        ss_res = np.sum(
            (y - predictions) ** 2
        )

        ss_tot = np.sum(
            (y - np.mean(y)) ** 2
        )

        if ss_tot == 0:
            return 0.0

        return 1.0 - (
            ss_res / ss_tot
        )

    def get_params(self, deep=True):
        """
        Return model parameters.

        This follows the scikit-learn style API.
        """

        return {
            "hidden_layer_size": self.hidden_layer_size,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "random_state": self.random_state,
        }

    def set_params(self, **params):
        """
        Set model parameters.

        This follows the scikit-learn style API.
        """

        for key, value in params.items():

            if not hasattr(self, key):
                raise ValueError(
                    f"Invalid parameter: {key}"
                )

            setattr(
                self,
                key,
                value,
            )

        return self
