# methods for inducing neural clustering by assigning particles to clusters and computing loss
import numpy as np

def compute_means(z, labels, d):
    """
    Compute class means and global mean.
    z: (n, num_steps, d)
    labels: (n)
    d: int dimension
    returns:
    class_means: dict
    global_mean: arr, global mean vector
    """
    n, num_steps, _ = z.shape
    classes = np.unique(labels)
    class_means = {}
    
    for cls in classes:
        class_indices = np.where(labels == cls)[0]
        if len(class_indices) == 0:
            raise ValueError(f"No particles found for class {cls}.")
        class_mean = np.mean(z[class_indices, 0, :], axis=0)
        class_means[cls] = class_mean
    
    global_mean = np.mean(z[:, 0, :], axis=0)
    
    return class_means, global_mean


def compute_unit_vectors(class_means, global_mean, d):
    """
    Compute unit vectors e_k for each class.
    return dict of class: unit vec
    """
    e_vectors = {}
    for cls, mean in class_means.items():
        direction = mean - global_mean
        norm = np.linalg.norm(direction)
        if norm == 0:
            raise ValueError(f"Class {cls} mean coincides with global mean.")
        e_vectors[cls] = direction / norm
    return e_vectors


def mean_inducing_V(I_d, class_means, global_mean, labels, d):
    """
    Initialize matrix V with inter-class repulsion.
    
    Args:
        I_d (np.ndarray): Identity matrix of shape (d, d).
        class_means (dict): Dictionary of class means.
        global_mean (np.ndarray): Global mean vector.
        labels (np.ndarray): Class labels of shape (n,).
        d (int): Number of dimensions.
    
    Returns:
        V (np.ndarray): Initialized V matrix.
    """
    e_vectors = compute_unit_vectors(class_means, global_mean, d)
    n = len(labels)
    K = len(class_means)
    
    V = I_d.copy()
    
    for cls, e_k in e_vectors.items():
        n_k = np.sum(labels == cls)
        V -= (n_k / n) * np.outer(e_k, e_k)
    
    return V