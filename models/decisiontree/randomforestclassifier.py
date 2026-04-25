from .randomforest import RandomForest

class RandomForestClassifier(RandomForest):
    def __init__(self, max_depth=None, min_samples_split=2):
        super().__init__(max_depth, min_samples_split, regression=False)
    