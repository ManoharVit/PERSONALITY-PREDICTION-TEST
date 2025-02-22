from DecisionTree import DecisionTree
import numpy as np

"""Random Forest classifier"""
class RandomForest:   
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []

    def fit(self, X, y):
        """Fits n_trees decision trees on bootstrap samples of the training data."""
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth,
                                min_samples_split=self.min_samples_split,
                                n_features=self.n_features)
            X_sample, y_sample = self._bootstrap_samples(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_samples(self, X, y):
        """Generate a bootstrap sample of the dataset."""
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def predict(self, X):
        """Predict the class for each sample in X based on majority voting from the trees."""
        tree_preds = np.array([tree.predict(X) for tree in self.trees]).T
        predictions = np.array([np.bincount(preds).argmax() for preds in tree_preds])
        return predictions
