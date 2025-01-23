import numpy as np

"""Gaussian Naive Bayes classifier implementation.""" 
class NaiveBayes:
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        # Initializing parameters for mean, variance, and prior probabilities
        self.mean_ = np.zeros((n_classes, n_features), dtype=np.float64)
        self.var_ = np.zeros((n_classes, n_features), dtype=np.float64)
        self.priors_ = np.zeros(n_classes, dtype=np.float64)
        
        for i, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.mean_[i, :] = X_c.mean(axis=0)
            self.var_[i, :] = X_c.var(axis=0)
            self.priors_[i] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        """Apply the Naive Bayes model to predict labels for a set of examples."""
        return np.array([self._predict(x) for x in X])

    def _predict(self, x):
        # Calculate the posterior probability for each class
        posteriors = [
            np.log(self.priors_[i]) + np.sum(self._log_pdf(i, x))
            for i in range(len(self.classes_))
        ]
        # Return the class with the highest posterior probability
        return self.classes_[np.argmax(posteriors)]

    def _log_pdf(self, class_idx, x):
        """Calculate the logarithm of the probability density function for a given class."""
        mean = self.mean_[class_idx]
        var = self.var_[class_idx]
        numerator = -((x - mean) ** 2) / (2 * var)
        denominator = np.log(np.sqrt(2 * np.pi * var))
        return numerator - denominator
