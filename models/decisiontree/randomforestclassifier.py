from .randomforest import RandomForest


class RandomForestClassifier(RandomForest):
    """
    Random Forest for classification tasks.

    Uses Gini impurity for splits and majority vote across trees.
    Feature subspace defaults to √n_features per the spec.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth=None,
        min_samples_split: int = 2,
        max_features: str = "sqrt",
        seed: int = None,
    ):
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            regression=False,
            min_samples_split=min_samples_split,
            max_features=max_features,
            seed=seed,
        )