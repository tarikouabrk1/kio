import numpy as np

class LogisticRegression:
    def __init__(self, lr, max_iters):
        self.lr = lr
        self.max_iters = max_iters

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):

        n, m = X.shape

        self.W = np.zeros(m)
        self.b = 0

        for _ in range(self.max_iters):

            z = X @ self.W + self.b
            y_pred = self.sigmoid(z)


            dw = (1 / n) * (X.T @ (y_pred - y))
            db = (1 / n) * np.sum(y_pred - y)

            self.W -= self.lr * dw
            self.b -= self.lr * db


    def predict(self, X):
        z = X @ self.W + self.b
        probs = self.sigmoid(z)
        return (probs > 0.5).astype(int)

    def predict_proba(self, X):
        z = X @ self.W + self.b
        return self.sigmoid(z)

    def transform(self, X: np.array):
        return self.predict(X)

    def get_W(self):
        return self.W
    def get_b(self):
        return self.b