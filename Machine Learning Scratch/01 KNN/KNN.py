import numpy as np

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        """Stores training data."""
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """Predicts labels for input data."""
        return np.array([self._predict(x) for x in X])

    def _predict(self, x):
        distances = self._euclidean_distance(x, self.X_train)
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices]
        return np.bincount(k_nearest_labels).argmax()

    def _euclidean_distance(self, x1, X2):
        """Calculate Euclidean distance between a single point and an array of points"""
        return np.linalg.norm(X2 - x1, axis=1)
