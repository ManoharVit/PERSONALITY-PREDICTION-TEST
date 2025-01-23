import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
from svm import SVM 

def accuracy(y_true, y_pred):
    """Calculate accuracy of predictions."""
    return np.mean(y_true == y_pred)

def visualize_svm(X, y, clf):
    """Visualize the decision boundaries of the SVM."""
    def get_hyperplane_value(x, w, b, offset):
        return (-w[0] * x + b + offset) / w[1]

    plt.scatter(X[:, 0], X[:, 1], marker='o', c=y)

    x0_1 = np.min(X[:, 0])
    x0_2 = np.max(X[:, 0])
    x1_1 = get_hyperplane_value(x0_1, clf.w, clf.b, 0)
    x1_2 = get_hyperplane_value(x0_2, clf.w, clf.b, 0)

    plt.plot([x0_1, x0_2], [x1_1, x1_2], 'y--')
    plt.show()

def main():
    X, y = datasets.make_blobs(n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=40)
    y = np.where(y == 0, -1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    clf = SVM(learning_rate=0.01, n_iters=1000)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    print("SVM classification accuracy", accuracy(y_test, predictions))
    visualize_svm(X_train, y_train, clf)

if __name__ == "__main__":
    main()
