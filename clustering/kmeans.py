"""Clustering models and utilities."""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA


def elbow_method(data: np.ndarray, max_k: int = 10, random_state: int = 42) -> tuple[list[int], list[float]]:
    """Compute inertia values for KMeans from k=1..max_k."""

    max_k = max(2, min(max_k, len(data)))
    ks = list(range(1, max_k + 1))
    inertias: list[float] = []

    for k in ks:
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        model.fit(data)
        inertias.append(float(model.inertia_))

    return ks, inertias


def train_kmeans(data: np.ndarray, n_clusters: int, random_state: int = 42) -> KMeans:
    """Fit and return a KMeans model."""

    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    model.fit(data)
    return model


def fit_pca(data: np.ndarray, n_components: int = 2) -> tuple[np.ndarray, PCA]:
    """Project data into PCA space."""

    pca = PCA(n_components=n_components, random_state=42)
    transformed = pca.fit_transform(data)
    return transformed, pca


def hierarchical_labels(data: np.ndarray, n_clusters: int) -> np.ndarray:
    """Fit agglomerative clustering and return cluster labels."""

    model = AgglomerativeClustering(n_clusters=n_clusters)
    return model.fit_predict(data)


def hierarchical_linkage(data: np.ndarray, method: str = "ward") -> np.ndarray:
    """Generate linkage matrix for dendrogram plotting."""

    return linkage(data, method=method)
