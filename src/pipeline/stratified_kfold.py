import numpy as np


class StratifiedKFoldScratch:
    """
    Stratified K-Fold cross-validation implemented from scratch.

    Each fold attempts to preserve the class distribution.
    """

    def __init__(
        self,
        n_splits=5,
        shuffle=False,
        random_state=None,
    ):
        if n_splits < 2:
            raise ValueError(
                "n_splits must be at least 2."
            )

        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y):
        """
        Generate train/test indices while preserving
        class proportions across folds.
        """

        X = np.asarray(X)
        y = np.asarray(y)

        if len(X) != len(y):
            raise ValueError(
                "X and y must have the same length."
            )

        rng = np.random.default_rng(
            self.random_state
        )

        classes = np.unique(y)

        fold_indices = [
            [] for _ in range(self.n_splits)
        ]

        # -----------------------------------------
        # Distribute samples of each class
        # -----------------------------------------

        for cls in classes:

            class_indices = np.where(
                y == cls
            )[0]

            if self.shuffle:
                rng.shuffle(class_indices)

            for i, index in enumerate(class_indices):
                fold_indices[
                    i % self.n_splits
                ].append(index)

        # -----------------------------------------
        # Generate train/test indices
        # -----------------------------------------

        all_indices = np.arange(len(X))

        for fold in range(self.n_splits):

            test_indices = np.array(
                fold_indices[fold],
                dtype=int
            )

            train_indices = np.setdiff1d(
                all_indices,
                test_indices
            )

            yield (
                train_indices,
                test_indices
            )