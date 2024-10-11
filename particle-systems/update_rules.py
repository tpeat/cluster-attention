import numpy as np

def update_A(attention_matrix, x, n, d):
    """
    Update transformation matrix A based on the current attention matrix and particle states.

    Args:
        attention_matrix (np.ndarray): Attention matrix of shape (n, n).
        x (np.ndarray): Current particle positions of shape (n, d).
        n (int): Number of particles.
        d (int): Dimensionality.

    Returns:
        np.ndarray: Updated transformation matrix A of shape (d, d).
    """
    attention_means = np.array([
        np.sum(attention_matrix[i, :, np.newaxis] * x, axis=0) 
        for i in range(n)
    ])
    A_new = np.zeros((d, d))
    for mean in attention_means:
        norm = np.linalg.norm(mean)
        if norm > 0:
            direction = mean / norm
            A_new += np.outer(direction, direction)
    A = A_new / n  # Normalize
    return A


def update_V(A, num_clusters):
    """
    Update transformation influence matrix V based on updated A.

    Args:
        A (np.ndarray): Updated transformation matrix of shape (d, d).
        num_clusters (int): Number of clusters.

    Returns:
        np.ndarray: Updated transformation influence matrix V of shape (d, d).
    """
    d = A.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    top_k = num_clusters
    V = eigenvectors[:, -top_k:] @ eigenvectors[:, -top_k:].T
    # Ensure V is positive semi-definite
    V = (V + V.T) / 2
    eigvals, _ = np.linalg.eigh(V)
    if eigvals.min() < 0:
        V += (0.01 - eigvals.min()) * np.eye(d)
    return V



# ------------------------- X updates --------------------------------

def euler_update(x_current, attention_matrix, V, dt):
    """
    Update particle states using Euler's method.

    Args:
        x_current (np.ndarray): Current particle positions of shape (n, d).
        attention_matrix (np.ndarray): Attention matrix of shape (n, n).
        V (np.ndarray): Transformation influence matrix of shape (d, d).
        dt (float): Time step.

    Returns:
        np.ndarray: Updated particle positions of shape (n, d).
    """
    n, d = x_current.shape
    dynamics = np.zeros((n, d))
    for i in range(n):
        dynamics[i] = np.sum(attention_matrix[i, :, np.newaxis] * (V @ x_current.T).T, axis=0)
    x_next = x_current + dt * dynamics
    return x_next


def rk4_update(x_current, attention_matrix, V, dt):
    """
    Update particle states using the 4th-order Runge-Kutta method.

    Args:
        x_current (np.ndarray): Current particle positions of shape (n, d).
        attention_matrix (np.ndarray): Attention matrix of shape (n, n).
        V (np.ndarray): Transformation influence matrix of shape (d, d).
        dt (float): Time step.

    Returns:
        np.ndarray: Updated particle positions of shape (n, d).
    """
    def dynamics(x):
        n, d = x.shape
        dxdt = np.zeros((n, d))
        for i in range(n):
            dxdt[i] = np.sum(attention_matrix[i, :, np.newaxis] * (V @ x.T).T, axis=0)
        return dxdt

    k1 = dynamics(x_current)
    k2 = dynamics(x_current + 0.5 * dt * k1)
    k3 = dynamics(x_current + 0.5 * dt * k2)
    k4 = dynamics(x_current + dt * k3)
    x_next = x_current + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return x_next


def update_state(x_current, attention_matrix, V, dt, method='Euler'):
    """
    Update particle states using the specified ODE solver.

    Args:
        x_current (np.ndarray): Current particle positions of shape (n, d).
        attention_matrix (np.ndarray): Attention matrix of shape (n, n).
        V (np.ndarray): Transformation influence matrix of shape (d, d).
        dt (float): Time step.
        method (str): ODE solver method ('Euler' or 'RK4').

    Returns:
        np.ndarray: Updated particle positions of shape (n, d).
    """
    if method == 'Euler':
        return euler_update(x_current, attention_matrix, V, dt)
    elif method == 'RK4':
        return rk4_update(x_current, attention_matrix, V, dt)
    else:
        raise ValueError(f"Unknown ODE solver method: {method}")
