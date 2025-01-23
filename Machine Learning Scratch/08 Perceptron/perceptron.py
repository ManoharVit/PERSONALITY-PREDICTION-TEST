import numpy as np

def unit_step_func(x):
    """Unit step activation function."""
    return np.where(x > 0, 1, 0)

"""Simple implementation of the Perceptron algorithm."""
    
class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = unit_step_func
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """Fit the perceptron model to the training data."""
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        y_ = np.where(y > 0, 1, 0)

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)
                update = self.lr * (y_[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        """Predict class labels for samples in X."""
        linear_output = np.dot(X, self.weights) + self.bias
        return self.activation_func(linear_output)
