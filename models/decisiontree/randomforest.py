from models.decisiontree.decisiontree import DecisionTree, DecisionStump, gini_impurity
import numpy as np
from dataframes import DataFrame

class RandomForest:
    def __init__(self, n_estimators=67, max_depth=None, regression=False, min_samples_split=2):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.regression = regression
        self.min_samples_split = min_samples_split
        self.trees = []

    def fit(self, X, y):
        for _ in range(self.n_estimators):
            indices = np.random.choice(len(X), len(X), replace=True)
            print(f"Training tree with indices: {indices}")
            X_sample = X[indices]
            y_sample = y[indices]
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split, regression=self.regression)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        predictions = [tree.predict(X) for tree in self.trees]
        if self.regression:
            return np.mean(predictions, axis=0)
        return [max(set(pred), key=pred.count) for pred in zip(*predictions)]
    
if __name__ == "__main__":
    data = DataFrame.DataFrame.load_csv("data.csv")
    model = RandomForest()
    model.fit(data, data["churn"])
    print(model.predict(data[0]))