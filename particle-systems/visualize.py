import numpy as np
import matplotlib.pyplot as plt

def visualize(x, A, attention_matrix, cluster_labels, colors, t, show=True):
    """
    Visualize the current state of the system.

    Args:
        x (np.ndarray): Current particle positions of shape (n, d).
        A (np.ndarray): Transformation matrix of shape (d, d).
        attention_matrix (np.ndarray): Attention matrix of shape (n, n).
        cluster_labels (np.ndarray): Cluster labels of shape (n,).
        colors (list): List of colors for each cluster.
        t (float): Current time.
        show (bool): Whether to display the plot immediately.
    """
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_aspect('equal', adjustable='box')
    
    # Plot transformed positions (A x)
    transformed_positions = (A @ x.T).T
    ax.scatter(
        transformed_positions[:, 0], 
        transformed_positions[:, 1],
        c=[colors[cluster_labels[i]] for i in range(x.shape[0])],
        alpha=0.7, 
        marker='o', 
        edgecolors='black', 
        label='A x'
    )
    
    # Plot original positions (x)
    ax.scatter(
        x[:, 0], 
        x[:, 1],
        c=[colors[cluster_labels[i]] for i in range(x.shape[0])],
        alpha=0.7, 
        marker='s', 
        edgecolors='black', 
        label='x'
    )
    
    # Draw connections based on attention
    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            if attention_matrix[i, j] > 1e-4 and i != j:
                ax.plot(
                    [transformed_positions[i, 0], x[j, 0]], 
                    [transformed_positions[i, 1], x[j, 1]],
                    linewidth=attention_matrix[i, j] * 1e-3,
                    color="black",
                    alpha=0.5
                )
    
    ax.set_title(f"t={t:.2f}", fontsize=16)
    ax.legend()
    ax.set_xticks([])
    ax.set_yticks([])
    if show:
        plt.show()
    else:
        plt.close()
