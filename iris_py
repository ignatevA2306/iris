import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

def elbow_method(data, max_k=10):
    inertia = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, max_iters=100)
        kmeans.fit(data)
        inertia.append(kmeans.inertia_)
    return inertia

class KMeans:
    def __init__(self, n_clusters=3, max_iters=100):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.centroids = None
        self.labels = None
        self.inertia_ = None

    def fit(self, X):
        n_samples, n_features = X.shape

        random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[random_indices]

        for i in range(self.max_iters):
            self.labels = self._assign_labels(X)

            old_centroids = self.centroids.copy()
            self._update_centroids(X)

            if np.allclose(old_centroids, self.centroids):
                break

        self.inertia_ = self._calculate_inertia(X)


    def _assign_labels(self, X):
        distances = np.linalg.norm(X[:, np.newaxis, :] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X):
        for cluster in range(self.n_clusters):
            cluster_points = X[self.labels == cluster]
            if len(cluster_points) > 0:
                self.centroids[cluster] = np.mean(cluster_points, axis=0)

    def predict(self, X):
        return self._assign_labels(X)

    def _calculate_inertia(self, X):
          inertia = 0
          for i in range(len(X)):
              centroid = self.centroids[self.labels[i]]
              inertia += np.sum((X[i] - centroid)**2)
          return inertia


inertia_values = elbow_method(X, max_k=10)

plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertia_values, marker='o')
plt.title('Метод локтя для определения оптимального количества кластеров')
plt.xlabel('Количество кластеров (k)')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()

optimal_k = 3


def plot_clusters(X, labels, centroids, iteration, feature1, feature2, feature_names):

    plt.figure(figsize=(8, 6))
    plt.title(f'Итерация {iteration + 1}')

    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

    for i in range(optimal_k):
        cluster_points = X[labels == i]
        plt.scatter(cluster_points[:, feature1], cluster_points[:, feature2], c=colors[i], label=f'Кластер {i + 1}')

    plt.scatter(centroids[:, feature1], centroids[:, feature2], marker='x', s=200, linewidths=3, color='black', label='Центроиды')
    plt.xlabel(feature_names[feature1])
    plt.ylabel(feature_names[feature2])
    plt.legend()
    plt.grid(True)
    plt.savefig(f'kmeans_iteration_{iteration + 1}.png')
    plt.close() # Закрываем текущий график, чтобы не отображался в конце выполнения


class KMeansWithVisualization(KMeans):

    def fit(self, X, feature1=0, feature2=1, feature_names=None):
        n_samples, n_features = X.shape

        random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[random_indices]

        for i in range(self.max_iters):
            self.labels = self._assign_labels(X)
            plot_clusters(X, self.labels, self.centroids, i, feature1, feature2, feature_names)
            old_centroids = self.centroids.copy()
            self._update_centroids(X)
            if np.allclose(old_centroids, self.centroids):
                break

        self.inertia_ = self._calculate_inertia(X)


kmeans_vis = KMeansWithVisualization(n_clusters=optimal_k, max_iters=10)
kmeans_vis.fit(X, feature1=0, feature2=1, feature_names = feature_names)

def plot_all_projections(X, labels, feature_names):

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()

    projections = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

    for i, (feature1, feature2) in enumerate(projections):
        ax = axes[i]

        for cluster in range(optimal_k):
            cluster_points = X[labels == cluster]
            ax.scatter(cluster_points[:, feature1], cluster_points[:, feature2], c=colors[cluster], label=f'Кластер {cluster + 1}' if i == 0 else "")

        ax.set_xlabel(feature_names[feature1])
        ax.set_ylabel(feature_names[feature2])
        ax.set_title(f'{feature_names[feature1]} vs {feature_names[feature2]}')
        ax.grid(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=optimal_k)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

plot_all_projections(X, kmeans_vis.labels, feature_names)
