import numpy as np

from src.pipeline.kfold import KFoldScratch


X = np.arange(20)

cv = KFoldScratch(
    n_splits=5,
    shuffle=False
)


print("\n========== K-FOLD TEST ==========\n")


for fold, (
    train_indices,
    test_indices
) in enumerate(
    cv.split(X),
    start=1
):

    print(f"Fold {fold}")

    print("Train indices:")
    print(train_indices)

    print("Test indices:")
    print(test_indices)

    print()