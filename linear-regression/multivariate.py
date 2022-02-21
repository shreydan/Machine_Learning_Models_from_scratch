import numpy as np


class LinearRegression:
    def __init__(self, learning_rate=1e-1, epochs=2500) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs

    def h_x(self, X, w):
        return np.dot(X, w)

    def cost(self, h_x, y):
        return (1 / (2 * self.m)) * np.sum(np.power(h_x - y, 2))

    def gradient_descent(self, X, y, w, h_x):

        # X:         m X (n+1)
        # X.T:      (n+1) X m
        # h_x - y:          m X 1
        # therefore X.T dot( h_x - y) giving us w.shape: (n+1) X 1

        dL = (2 / self.m) * np.dot(X.T, (h_x - y))
        w -= self.learning_rate * dL
        return w

    def fit(self, X, y):
        # X: m X n matrix
        # y: m X 1 matrix

        self.m = X.shape[0]  # no. of training samples
        self.n = X.shape[1]  # no. of features

        ones = np.ones(shape=X.shape[0])  # 1 X m matrix
        X = np.c_[ones, X]
        self.w = np.zeros(shape=self.n + 1)  # n+1 X 1

        for _ in range(self.epochs):

            h_x = self.h_x(X, self.w)
            cost = self.cost(h_x, y)
            self.w = self.gradient_descent(X, y, self.w, h_x)
            self.loss = cost

    def predict(self, X):
        ones = np.ones(shape=X.shape[0])
        X = np.c_[ones, X]
        return self.h_x(X, self.w)
