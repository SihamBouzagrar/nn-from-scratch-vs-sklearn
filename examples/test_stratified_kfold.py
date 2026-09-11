import numpy as np

from src.pipeline.stratified_kfold import (
    StratifiedKFoldScratch
)


# 70 % class 0
# 30 % class 1

y = np.array([
    0, 0, 0, 0, 0, 0, 0,
    1, 1, 1
])

X = np.arange(
    len(y)
)


cv = StratifiedKFoldScratch(
    n_splits=3,
    shuffle=True,
    random_state=42
)


print("\n========== STRATIFIED K-FOLD ==========\n")


for fold, (
    train_idx,
    test_idx
) in enumerate(
    cv.split(X, y),
    start=1
):

    print(f"Fold {fold}")

    print("Test indices :", test_idx)

    print(
        "Test labels  :",
        y[test_idx]
    )

    print(
        "Class 0      :",
        np.sum(y[test_idx] == 0)
    )

    print(
        "Class 1      :",
        np.sum(y[test_idx] == 1)
    )

    print()