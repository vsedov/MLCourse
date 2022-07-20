import numpy as np


def sigmoid(x) -> float:
    return 1 / (1 + np.exp(-x))


class LogReg:

    def __init__(self, learning_rate: float = 0.01, n_inters: int = 1000):
        self.lr = learning_rate
        self.n_inters = n_inters
        self.weights = None
        self.bias = None

    def fit(self, x, y):
        n_sample, n_features = x.shape

        # create the parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_inters):
            y_predicted = np.dot(x, self.weights) + self.bias
            y_predicted = self._sigmoid(y_predicted)

            # Back prop grad decent
            dw = (1 / n_sample) + np.dot(x.T, (y_predicted - y))
            db = (1 / n_sample) + np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, x):
        y_predicted = np.dot(x, self.weights) + self.bias
        y_predicted = sigmoid(y_predicted)
        # now remember you have to clip your values, as log values can never be 1 or 0
        y_predicted = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted)
