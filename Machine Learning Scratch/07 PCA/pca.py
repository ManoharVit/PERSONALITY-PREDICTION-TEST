import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        """
        Fit the model with X by finding the principal components.
        """
        # Centering the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # Compute the covariance matrix
        cov_matrix = np.cov(X_centered.T)

        # Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        eigenvectors = eigenvectors.T

        # Sorting eigenvectors by decreasing eigenvalues
        idxs = np.argsort(eigenvalues)[::-1]
        self.components = eigenvectors[idxs[:self.n_components]]

    def transform(self, X):
        """
        Apply the dimensionality reduction on X.
        """
        X_centered = X - self.mean
        return np.dot(X_centered, self.components.T)
