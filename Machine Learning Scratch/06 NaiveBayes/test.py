# test.py
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from NaiveBayes import NaiveBayes  

def accuracy(y_true, y_pred):
    """Calculate the accuracy of the predictions."""
    return np.mean(y_true == y_pred)

def main():
    # Load data
    X, y = datasets.make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=123)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    # Initialize and fit NaiveBayes model
    nb = NaiveBayes()
    nb.fit(X_train, y_train)
    predictions = nb.predict(X_test)

    print("Naive Bayes classification accuracy:", accuracy(y_test, predictions))

if __name__ == "__main__":
    main()
