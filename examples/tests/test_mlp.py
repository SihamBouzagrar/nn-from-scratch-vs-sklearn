import numpy as np

from sklearn.datasets import make_moons

from src.engine.mlp_classifier import (
    MLPClassifierScratch
)


def test_mlp_output_shape():

    X, y = make_moons(
        n_samples=100,
        noise=0.2,
        random_state=42
    )

    model = MLPClassifierScratch(
        hidden_layer_size=8,
        learning_rate=0.05,
        epochs=100
    )

    model.fit(X, y)

    predictions = model.predict(X)

    assert predictions.shape == y.shape