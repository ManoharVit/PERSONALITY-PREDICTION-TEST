import sys 
  
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from PCA import PCA
import os
import pickle

def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def load_data():
    """Load a dataset to fit the PCA."""
    data = datasets.load_iris()
    X = data.data
    y = data.target
    return X, y

def train(X):
    """Train PCA model on the dataset."""
    pca = PCA(n_components=2)
    pca.fit(X)
    return pca

def main():
    X, y = load_data()
    pca_model = train(X)

    if not os.path.exists('models'):
        os.makedirs('models')
    save_model(pca_model, 'models/pca_model.pkl')

    # Output some basic information about the components
    print("PCA Components:")
    print(pca_model.components)

    # Plot of dataset projected onto the first two principal components
    X_projected = pca_model.transform(X)
    plt.scatter(X_projected[:, 0], X_projected[:, 1], c=y, cmap=plt.cm.get_cmap('viridis', 3))
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    main()
