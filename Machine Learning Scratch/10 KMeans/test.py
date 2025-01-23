import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from kmeans import KMeans  

""" Test the KMeans algorithm with specified number of clusters """
def test_kmeans(data, n_clusters):
    # Initialize KMeans
    kmeans = KMeans(K=n_clusters, max_iters=150, plot_steps=False)
    # Fit and predict clusters
    y_pred = kmeans.predict(data)
    
    # Calculate silhouette score for evaluation
    score = silhouette_score(data, y_pred)
    print(f"Silhouette Score: {score:.3f}")
    
    # Visualization
    visualize_clusters(data, y_pred, kmeans.centroids)

def visualize_clusters(X, labels, centroids):
    """ Visualize the data points and the centroids of clusters """
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k', alpha=0.7)
    plt.scatter(centroids[:, 0], centroids[:, 1], s=250, c='red', marker='X', edgecolor='black', label='Centroids')
    plt.colorbar(scatter)
    plt.title('KMeans Clustering Results')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    plt.show()


# Test KMeans with 3 clusters
def main():
    # Generate synthetic data
    X, _ = make_blobs(centers=3, n_samples=300, n_features=2, shuffle=True, random_state=42)
    test_kmeans(X, 3)

if __name__ == "__main__":
    main()
