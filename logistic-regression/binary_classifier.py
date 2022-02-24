import numpy as np


class LogisticRegression:
    def __init__(self, learning_rate=1e-2, epochs=10000) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.decision_boundary = 0.5

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def y_pred(self, X, w, b):

        # X : m X n
        # w : n X 1
        # X.w + b: m X 1
        # y_pred: m X 1

        return self.sigmoid(np.dot(X, w) + b)

    def cost(self, y_pred, y):

        # y_pred: m X 1
        # y: m X 1
        # cost: scalar

        # * : element-wise matrix multiplication

        positive_side = y * np.log(y_pred)
        negative_side = (1 - y) * np.log(1 - y_pred)

        return (-1 / self.m) * np.sum(positive_side + negative_side)

    def gradient_descent(self, X, y, y_pred, w, b):

        # X: m X n
        # X.T: n X m
        # y or y_pred: m X 1
        # X.T dot (y_pred-y) = n X 1 == W.shape : n X 1
        # b: scalar

        dW = (1 / self.m) * np.dot(X.T, (y_pred - y))
        db = (1 / self.m) * np.sum(y_pred - y)

        w -= self.learning_rate * dW
        b -= self.learning_rate * db

        return w, b

    def fit(self, X, y):

        # X: m X n
        # y: m X 1

        # m: no. of samples
        # n: no. of features
        self.m, self.n = X.shape

        # weights: n X 1
        # bias: scalar

        self.weights = np.zeros(shape=self.n)
        self.bias = 0

        for _ in range(self.epochs):

            y_pred = self.y_pred(X, self.weights, self.bias)
            self.weights, self.bias = self.gradient_descent(
                X, y, y_pred, self.weights, self.bias
            )
            cost = self.cost(y_pred, y)

            self.loss = cost

    def predict(self, X):

        y_pred = self.y_pred(X, self.weights, self.bias)
        y_pred_labels = y_pred >= self.decision_boundary
        y_pred_labels = y_pred_labels.astype(int)

        return y_pred_labels
