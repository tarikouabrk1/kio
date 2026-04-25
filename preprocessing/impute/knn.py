import numpy as np
from math_utils.statistics import distance, mean, majority_vote, is_missing
from dataframes.DataFrame import  DataFrame

class KNNImputer:
    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors

    def fit(self, X):

        X = X.copy()
        N = X.get_numerical()
        C = X.get_object()

        n_rows, n_cols = X.format()

        null_lines = []
        non_null_lines = []
        for i in range(n_rows):
            if any(is_missing(v) for v in X[i]):
                null_lines.append((i, X[i]))
            else:
                non_null_lines.append((i, X[i]))

        for idx, line in null_lines:

            Distances = []

            for j, arr in non_null_lines:
                Distances.append((j, distance(line, arr)))

            Distances = sorted(Distances, key=lambda x: x[1])
            usefullDistances = Distances[:self.n_neighbors]

            neighbors_idx = [i for i, _ in usefullDistances]

            for col in range(n_cols):

                if is_missing(line[col]):

                    values = []

                    for j in neighbors_idx:
                        if not is_missing(X[j][col]):
                            values.append(X[j][col])

                    if type(line[col]) == np.float64:
                        X[idx][col] = mean(values)[0]
                    else:
                        X[idx][col] = majority_vote(values)[0]


        self.X_ = X
        return self

    def transform(self, X: np.array) -> np.array:
        return self.X_

    def fit_transform(self, X: np.array) -> np.array:
        self.fit(X)
        return self.transform(X)


