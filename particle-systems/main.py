#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: borjangeshkovski
Modified by: [Your Name]
"""

import numpy as np
import matplotlib.pyplot as plt

# Import the modular functions (assuming they are in the same script or appropriately imported)
# from modular_functions import (
#     initialize_particles, initialize_A, initialize_V, update_A, update_V,
#     visualize, update_state
# )

# Parameters
n = 20          # Total number of particles
T = 10          # Total simulation time
dt = 0.1        # Time step
num_steps = int(T/dt) + 1
d = 2           # Dimensionality (2D)

# Define cluster centers (modifiable for different experiments)
cluster_centers = np.array([[4, 4], [-4, -4], [4, -4], [-4, 4]])  # Four clusters
num_clusters = cluster_centers.shape[0]
particles_per_cluster = n // num_clusters

# Initialize particles
x0, cluster_labels = initialize_particles(cluster_centers, particles_per_cluster, noise=0.5)
x = np.zeros((n, num_steps, d))
x[:, 0, :] = x0
integration_time = np.linspace(0, T, num_steps)

# Initialize transformation matrices A and V
A = initialize_A(cluster_centers, num_clusters)
V = initialize_V(A, num_clusters)

# Define colors for visualization (extend as needed)
colors = ["#a8deb5", "#a5d8ff", "#ffb3c1", "#c1caff"]  # For four clusters

# Simulation Loop
for l, t in enumerate(integration_time):
    if l < num_steps - 1:
        # Compute Attention Matrix
        attention_matrix = np.array([
            [
                1 / np.sum([
                    np.exp(np.dot(A @ x[i, l], x[k, l] - x[j, l]))
                    for k in range(n)
                ]) 
                for j in range(n)
            ] 
            for i in range(n)
        ])
        rank = np.linalg.matrix_rank(attention_matrix)
        print(f"Time {t:.2f}: Attention Matrix Rank = {rank}")
        
        # Update Transformation Matrix A (modular, can be swapped out)
        A = update_A(attention_matrix, x[:, l, :], n, d)
        
        # Update Transformation Influence Matrix V (modular, can be swapped out)
        V = update_V(A, num_clusters)
        
        # Visualization at every 0.5 time units
        if np.isclose(t % 0.5, 0, atol=1e-8):
            visualize(
                x=x[:, l, :],
                A=A,
                attention_matrix=attention_matrix,
                cluster_labels=cluster_labels,
                colors=colors,
                t=t,
                show=True
            )
        
        # State Update using the desired ODE solver
        # Choose between 'Euler' and 'RK4'
        x[:, l + 1, :] = update_state(
            x_current=x[:, l, :],
            attention_matrix=attention_matrix,
            V=V,
            dt=dt,
            method='Euler'  # Change to 'RK4' for Runge-Kutta
        )
