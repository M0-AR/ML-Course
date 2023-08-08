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

# -----------
import numpy as np


def k_means_iterations(points, centroids, num_iterations):
    # Initialize cost
    cost = 0
    for _ in range(num_iterations):
        # Compute distances from points to each centroid
        distances = np.array([np.abs(points - c) for c in centroids])

        # Assign each point to nearest centroid
        assignments = np.argmin(distances, axis=0)

        # Update centroids as mean of assigned points
        centroids = np.array([np.mean(points[assignments == i]) for i in range(len(centroids))])

        # Compute cost (sum of squared distances) based on updated centroids
        cost = np.sum([np.sum((points[assignments == i] - centroids[i]) ** 2) for i in range(len(centroids))])

    return cost


points = np.array([2, 5, 8, 12, 13])
centroids = np.array([4, 10])
num_iterations = 1


cost = k_means_iterations(points, centroids, num_iterations)
print('\n\n')
print('The total cost after the first iteration of K-means is {}'.format(cost))
print('\n\n')

# ---

import numpy as np


def k_means_update(points, centroids, tolerance=1e-5, max_iter=100):
    # Continue the loop until either the maximum number of iterations is reached
    # or the change in centroids is less than the tolerance
    for _ in range(max_iter):
        # Compute distances from points to each centroid
        distances = np.array([np.abs(points - c) for c in centroids])

        # Assign each point to nearest centroid
        assignments = np.argmin(distances, axis=0)

        # Compute new centroids as mean of assigned points
        new_centroids = np.array([np.mean(points[assignments == i]) for i in range(len(centroids))])

        # Check if change in centroids is less than tolerance
        if np.all(np.abs(new_centroids - centroids) < tolerance):
            break

        # Update centroids
        centroids = new_centroids

    return centroids, assignments


def k_means_cost(points, centroids, assignments):
    # Compute cost (sum of squared distances) based on updated centroids
    cost = np.sum([np.sum((points[assignments == i] - centroids[i]) ** 2) for i in range(len(centroids))])

    return cost


# Input data
points = np.array([2, 5, 8, 12, 13])
initial_centroids = np.array([4, 10])

# Run K-means to update centroids
final_centroids, assignments = k_means_update(points, initial_centroids)

# Calculate cost
cost = k_means_cost(points, final_centroids, assignments)

# Print the results
print(f'The final centroids are: {final_centroids}')
print(f'The assignments are: {assignments}')
print(f'The total cost after the first iteration of K-means is: {cost}')
