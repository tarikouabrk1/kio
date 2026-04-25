from .randomforest import RandomForest


class RandomForestRegression(RandomForest):
    """
    Random Forest for regression tasks.

    Uses MSE variance reduction for splits and averages leaf
    predictions across all trees.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth=None,
        min_samples_split: int = 2,
        max_features="third",
        seed: int = None,
    ):
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            regression=True,
            min_samples_split=min_samples_split,
            max_features=max_features,
            seed=seed,
        )