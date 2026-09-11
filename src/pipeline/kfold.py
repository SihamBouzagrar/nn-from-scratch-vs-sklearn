import numpy as np


class KFoldScratch:
    """
    Simple K-Fold cross-validation implemented from scratch.
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

    def split(self, X):
        """
        Generate train/test indices for each fold.
        """

        X = np.asarray(X)

        n_samples = len(X)

        indices = np.arange(n_samples)

        if self.shuffle:
            rng = np.random.default_rng(
                self.random_state
            )

            rng.shuffle(indices)

        fold_sizes = np.full(
            self.n_splits,
            n_samples // self.n_splits,
            dtype=int,
        )

        fold_sizes[
            : n_samples % self.n_splits
        ] += 1

        current = 0

        for fold_size in fold_sizes:

            start = current
            stop = current + fold_size

            test_indices = indices[
                start:stop
            ]

            train_indices = np.concatenate([
                indices[:start],
                indices[stop:],
            ])

            yield (
                train_indices,
                test_indices
            )

            current = stop