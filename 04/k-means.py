"""
02450ex_Fall2021_sol-1.pdf
Question 20
"""
import numpy as np

# Original dataset
x = np.array([0.4, 1.7, 3.7, 4.6])

# Number of clusters
K = 3

# First centroid
centroids = [1.7]

# Find the rest of the centroids
for _ in range(K - 1):
    # Compute distances from current centroids
    distances = [min(abs(xi - mu) for mu in centroids) for xi in x]

    # Select as the next centroid the point with the maximum distance
    next_centroid = x[np.argmax(distances)]
    centroids.append(next_centroid)

# Print the centroids
print("The centroids are:", centroids)