import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from svm import SVM 
import pickle

def save_model(model, filename="svm_model.pkl"):
    """Save the trained model to disk."""
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def load_data():
    """Load and prepare the data."""
    X, y = datasets.make_blobs(n_samples=100, n_features=2, centers=2, cluster_std=1.5, random_state=42)
    y = np.where(y == 0, -1, 1)  # Convert labels to -1, 1 as required by SVM
    return X, y

def train(X, y):
    """Train the SVM model."""
    clf = SVM(learning_rate=0.01, lambda_param=0.01, n_iters=1000)
    clf.fit(X, y)
    return clf

def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = train(X_train, y_train)
    save_model(model)
    print("Model trained and saved to 'svm_model.pkl'")

if __name__ == "__main__":
    main()
