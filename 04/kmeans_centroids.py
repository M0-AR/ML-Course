"""
Calculates distances between data points and centroids in two iterations of a clustering algorithm (such as k-means). Here are the main points:

Functions for Calculations and Printing:

calculate_distances: This function takes the data points and centroids as input and returns the absolute distances between each data point and each centroid.
print_distances: This function takes the iteration number and the calculated distances and prints them nicely to the console.
Data Points and Centroids: The data points and initial centroids are the same as before, and they are clearly defined in the code.

First Iteration: You calculate and print the distances between the data points and the initial centroids using the two defined functions. The distances represent how far each data point is from each centroid.

Second Iteration: You again calculate and print the distances between the data points and the centroids, but the centroids are the same as in the first iteration.

Lack of Full k-means Algorithm: While the code clearly calculates and prints the distances, it doesn't implement the
full k-means algorithm. Specifically, it lacks the steps to assign each point to the closest cluster and to update
the centroids based on the mean of the points in each cluster.

In summary, the code demonstrates a clear and functional way to calculate distances between data points and centroids for two iterations but does not proceed with clustering the data or updating the centroids.
"""
import numpy as np

"""This code is organized to clearly illustrate the process of calculating distances in two iterations. By utilizing 
functions and clear naming conventions, the code becomes more maintainable and easier to follow. The comments and 
docstrings help to provide context for what the code is doing at each step. """


def calculate_distances(data, centroids):
    """
    Calculate the absolute distances between data points and centroids.

    Args:
        data (list): The data points.
        centroids (list): The centroids.

    Returns:
        list: A list of distances for each centroid.
    """
    distances = [np.absolute(np.subtract(data, centroid)) for centroid in centroids]
    return distances


def print_distances(iteration, distances):
    """
    Prints the distances for the given iteration.

    Args:
        iteration (int): The iteration number.
        distances (list): The distances for each centroid.
    """
    print(f"Iteration {iteration}")
    for i, dist in enumerate(distances, 1):
        print(f"Dist {i}: ", dist)
    print()


if __name__ == "__main__":
    # Data points
    data = [-2.1, -1.7, -1.5, -0.4, 0, 0.6, 0.8, 1, 1.1]

    # Initial centroids
    centroids_init = [-2.1, -1.7, -1.5]

    # First iteration distances
    distances_iteration_1 = calculate_distances(data, centroids_init)
    print_distances(1, distances_iteration_1)

    # Second iteration centroids
    centroids_iteration_2 = [-2.1, -1.7, -1.5]

    # Second iteration distances
    distances_iteration_2 = calculate_distances(data, centroids_iteration_2)
    print_distances(2, distances_iteration_2)

"""
The output represents the distances between data points and centroids for two iterations of a clustering algorithm (e.g., k-means). Since the centroids don't change between the iterations, the distances remain the same in both iterations. Here's an explanation for each part of the output:

Iteration 1:
    Dist 1: Distances between data points and the first centroid (-2.1).
    Dist 2: Distances between data points and the second centroid (-1.7).
    Dist 3: Distances between data points and the third centroid (-1.5).
    
Iteration 2: As the centroids haven't changed, the distances remain the same in the second iteration.

Distance Values: The distances are computed as the absolute difference between each data point and each centroid. For example, the distance between the first data point (-2.1) and the first centroid (-2.1) is |-2.1 - (-2.1)| = 0.

Lack of Clustering: The code only calculates the distances between the points and centroids but doesn't assign the points to clusters or update the centroids. As a result, the distances remain the same between iterations.

Implications for k-means: In a complete k-means algorithm, the next step after calculating distances would be to assign each point to the closest cluster (i.e., centroid) and then update the centroids based on the mean of the points in each cluster. Since these steps are missing in the code, the centroids and distances remain unchanged across iterations.

In summary, the output shows the absolute distances between data points and three static centroids across two iterations. Since the centroids don't change, neither do the distances. To perform clustering, additional steps would be required to assign points to clusters and update the centroids.
"""
