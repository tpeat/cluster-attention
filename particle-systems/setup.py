# methods to initialize particles, A, V

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def initialize_particles_baseline(n, d, num_steps):
    x0 = np.random.uniform(low=-1, high=1, size=(n,d))
    x = np.zeros(shape=(n,num_steps, d))
    x[:, 0, :] = x0
    return x


def initialize_particles_around_clusters(cluster_centers, particles_per_cluster, noise):
    """
    Initialize particle positions around specified cluster centers with Gaussian noise.

    Args:
        cluster_centers (np.ndarray): Array of shape (num_clusters, d) specifying cluster centers.
        particles_per_cluster (int): Number of particles per cluster.
        noise (float): Standard deviation of Gaussian noise.

    Returns:
        np.ndarray: Initialized particle positions of shape (n, d).
        np.ndarray: Cluster labels of shape (n,).
    """
    num_clusters, d = cluster_centers.shape
    n = particles_per_cluster * num_clusters
    x0 = np.vstack([
        cluster_centers[i] + noise * np.random.randn(particles_per_cluster, d)
        for i in range(num_clusters)
    ])
    cluster_labels = np.array([
        i for i in range(num_clusters) for _ in range(particles_per_cluster)
    ])
    return x0, cluster_labels

def initialize_A_baseline(d):
    return np.eye(d)

def initialize_A_on_clusters(cluster_centers, num_clusters):
    """
    Initialize the transformation matrix A based on cluster centers.

    Args:
        cluster_centers (np.ndarray): Array of shape (num_clusters, d) specifying cluster centers.
        num_clusters (int): Number of clusters.

    Returns:
        np.ndarray: Initialized transformation matrix A of shape (d, d).
    """
    d = cluster_centers.shape[1]
    A = np.zeros((d, d))
    for c in range(num_clusters):
        direction = cluster_centers[c] / np.linalg.norm(cluster_centers[c])
        A += np.outer(direction, direction)
    A = A / num_clusters  # Normalize
    return A


def initailize_V_baseline(d):
    V = 2 * np.random.rand(d, d) - np.ones((d,d))
    V = np.matmul(V, V.T) # makes V SPD
    return V

def initialize_V_on_clusters(A, num_clusters):
    """
    Initialize the transformation influence matrix V based on matrix A.

    Args:
        A (np.ndarray): Transformation matrix of shape (d, d).
        num_clusters (int): Number of clusters.

    Returns:
        np.ndarray: Initialized transformation influence matrix V of shape (d, d).
    """
    d = A.shape[0]
    V = np.zeros((d, d))
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    top_k = num_clusters
    V = eigenvectors[:, -top_k:] @ eigenvectors[:, -top_k:].T
    # Ensure V is positive semi-definite
    V = (V + V.T) / 2
    eigvals, _ = np.linalg.eigh(V)
    if eigvals.min() < 0:
        V += (0.01 - eigvals.min()) * np.eye(d)
    return V

def init_particles(exp_name):
    if exp_name == "baseline":
        return initialize_particles_baseline
    elif exp_name == "cluster_centers":
        return initialize_particles_around_clusters

def init_A(exp_name):
    if exp_name == "baseline":
        return initialize_A_baseline
    elif exp_name == "cluster_centers":
        return initialize_A_on_clusters

def init_V(exp_name):
    if exp_name == 'baseline':
        return initailize_V_baseline
    elif exp_name == "cluster_centers":
        return initialize_V_on_clusters