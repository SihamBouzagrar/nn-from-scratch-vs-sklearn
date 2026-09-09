import numpy as np


class Perceptron:
    """
    Perceptron binary classifier implemented from scratch with NumPy.
    """

    def __init__(self, learning_rate=0.01, epochs=100):
       
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.weights = None
        self.bias = None
        self.errors_ = []

    def _initialize_parameters(self, n_features):
        """
        Initialize model parameters.
        """
        self.weights = np.zeros(n_features)
        self.bias = 0.0

    def _step_function(self, z):
        """
        Binary step activation function.
        """
        return 1 if z >= 0 else 0

    def _predict_one(self, x):
        """
        Predict the class of a single sample.
        """
        z = np.dot(x, self.weights) + self.bias
        return self._step_function(z)

    def fit(self, X, y):
        """
        Train the Perceptron.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        n_samples, n_features = X.shape

        self._initialize_parameters(n_features)

        for epoch in range(self.epochs):
            errors = 0

            for x_i, y_i in zip(X, y):
                y_pred = self._predict_one(x_i)

                error = y_i - y_pred

                self.weights += (
                    self.learning_rate * error * x_i
                )

                self.bias += self.learning_rate * error

                if error != 0:
                    errors += 1

            self.errors_.append(errors)

            if errors == 0:
                break

        return self

    def predict(self, X):
        """
        Predict classes for multiple samples.
        """
        X = np.asarray(X)

        return np.array([
            self._predict_one(x)
            for x in X
        ])

    def score(self, X, y):
        """
        Return classification accuracy.
        """
        y = np.asarray(y)
        predictions = self.predict(X)

        return np.mean(predictions == y)