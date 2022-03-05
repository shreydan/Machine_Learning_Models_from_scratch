import numpy as np


class KNNClassifier:
    def __init__(self, n_neighbors=5):
        self.k = n_neighbors

    def euclidean_distance(self, x_train, x_test):
        # m = no. of distances i.e. training points for each test point
        m = x_train.shape[0]
        n = x_test.shape[0]  # each test point
        distances = np.zeros(shape=(n, m))
        for i in range(n):
            distances[i] = np.sqrt(np.sum((x_train - x_test[i]) ** 2, axis=1))
        return distances

    def predict(self, x_test):
        m = x_test.shape[0]  # no. of examples in x_test
        distances = self.euclidean_distance(self.x_train, x_test)
        sorted_args = np.argsort(distances, axis=1)

        closest_labels = self.y_train[sorted_args[:, : self.k]]
        y_preds = np.zeros(shape=m)

        for r in range(m):
            y_preds[r] = np.bincount(closest_labels[r]).argmax()

        return y_preds.astype(int)

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
