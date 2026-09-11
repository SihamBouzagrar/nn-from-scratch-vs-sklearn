import numpy as np


def cross_validate(
    model_class,
    X,
    y,
    cv,
    model_params=None,
    scaler_class=None,
):
    """
    Generic cross-validation utility.

    Parameters
    ----------
    model_class : class
        Model class with fit() and predict().

    X : array-like
        Features.

    y : array-like
        Target.

    cv : object
        Object exposing split(X, y).

    model_params : dict, optional
        Parameters passed to the model.

    scaler_class : class, optional
        Preprocessing scaler such as StandardScaler.
    """

    X = np.asarray(X)
    y = np.asarray(y)

    if model_params is None:
        model_params = {}

    scores = []

    for train_idx, test_idx in cv.split(X, y):

        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        # -------------------------------
        # Preprocessing
        # -------------------------------

        if scaler_class is not None:

            scaler = scaler_class()

            X_train = scaler.fit_transform(
                X_train
            )

            X_test = scaler.transform(
                X_test
            )

        # -------------------------------
        # New model
        # -------------------------------

        model = model_class(
            **model_params
        )

        # -------------------------------
        # Training
        # -------------------------------

        model.fit(
            X_train,
            y_train
        )

        # -------------------------------
        # Evaluation
        # -------------------------------

        predictions = model.predict(
            X_test
        )

        score = np.mean(
            predictions == y_test
        )

        scores.append(score)

    scores = np.asarray(scores)

    return {
        "scores": scores,
        "mean": scores.mean(),
        "std": scores.std(),
    }