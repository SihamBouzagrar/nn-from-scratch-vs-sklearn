import itertools
import numpy as np


class GridSearchScratch:
    """
    Simple Grid Search implementation from scratch.

    The class evaluates every hyperparameter combination
    using cross-validation.
    """

    def __init__(
        self,
        model_class,
        param_grid,
        cv,
        scaler_class=None,
    ):
        self.model_class = model_class
        self.param_grid = param_grid
        self.cv = cv
        self.scaler_class = scaler_class

        self.cv_results_ = []
        self.best_params_ = None
        self.best_score_ = None
        self.best_model_ = None

    def _generate_param_combinations(self):
        """
        Generate all hyperparameter combinations.
        """

        keys = list(self.param_grid.keys())

        values = [
            self.param_grid[key]
            for key in keys
        ]

        combinations = []

        for combination in itertools.product(*values):

            params = dict(
                zip(keys, combination)
            )

            combinations.append(params)

        return combinations

    def fit(self, X, y):

        X = np.asarray(X)
        y = np.asarray(y)

        combinations = (
            self._generate_param_combinations()
        )

        best_score = -np.inf
        best_params = None

        for params in combinations:

            fold_scores = []

            for train_idx, test_idx in self.cv.split(X, y):

                X_train = X[train_idx]
                X_test = X[test_idx]

                y_train = y[train_idx]
                y_test = y[test_idx]

                # ----------------------------
                # Preprocessing
                # ----------------------------

                if self.scaler_class is not None:

                    scaler = self.scaler_class()

                    X_train = scaler.fit_transform(
                        X_train
                    )

                    X_test = scaler.transform(
                        X_test
                    )

                # ----------------------------
                # New model
                # ----------------------------

                model = self.model_class(
                    **params
                )

                # ----------------------------
                # Train
                # ----------------------------

                model.fit(
                    X_train,
                    y_train
                )

                # ----------------------------
                # Evaluate
                # ----------------------------

                predictions = model.predict(
                    X_test
                )

                score = np.mean(
                    predictions == y_test
                )

                fold_scores.append(score)

            fold_scores = np.asarray(
                fold_scores
            )

            mean_score = fold_scores.mean()

            std_score = fold_scores.std()

            result = {
                "params": params,
                "mean_test_score": mean_score,
                "std_test_score": std_score,
                "fold_scores": fold_scores,
            }

            self.cv_results_.append(result)

            print(
                f"Params: {params}"
            )

            print(
                f"Mean score: {mean_score:.4f}"
            )

            print(
                f"Std: {std_score:.4f}"
            )

            print("-" * 50)

            # ----------------------------
            # Best model
            # ----------------------------

            if mean_score > best_score:

                best_score = mean_score
                best_params = params

        self.best_score_ = best_score
        self.best_params_ = best_params

        # Train final best model on all data
        X_final = X

        if self.scaler_class is not None:

            self.scaler_ = self.scaler_class()

            X_final = self.scaler_.fit_transform(
                X_final
            )

        self.best_model_ = self.model_class(
            **self.best_params_
        )

        self.best_model_.fit(
            X_final,
            y
        )

        return self

    def predict(self, X):
        """
        Predict using the best model.
        """

        X = np.asarray(X)

        if self.scaler_class is not None:
            X = self.scaler_.transform(X)

        return self.best_model_.predict(X)