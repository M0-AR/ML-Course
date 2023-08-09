import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt


def hierarchical_clustering(distance_matrix, method='single'):
    """
    Perform hierarchical clustering and plot the dendrogram.

    Parameters:
    - distance_matrix: 2D numpy array representing the pairwise distance between points.
    - method: string specifying the linkage method ('single', 'average', or 'complete').
    """

    # The linkage function for hierarchical clustering in scipy requires condensed distance matrix.
    condensed_matrix = []
    for i in range(len(distance_matrix)):
        for j in range(i + 1, len(distance_matrix)):
            condensed_matrix.append(distance_matrix[i][j])

    # Perform hierarchical clustering
    linked = linkage(condensed_matrix, method=method)

    # Plot the dendrogram
    plt.figure(figsize=(10, 7))
    dendrogram(linked, labels=["O" + str(i + 1) for i in range(len(distance_matrix))])
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Data points')
    plt.ylabel('Distance')
    plt.show()


# Example
distance_matrix = np.array([
    [0, 2.39, 1.73, 0.96, 3.46, 4.07, 4.27, 5.11],
    [2.39, 0, 1.15, 1.76, 2.66, 5.36, 3.54, 4.79],
    [1.73, 1.15, 0, 1.52, 3.01, 4.66, 3.77, 4.90],
    [0.96, 1.76, 1.52, 0, 2.84, 4.25, 3.80, 4.74],
    [3.46, 2.66, 3.01, 2.84, 0, 4.88, 1.41, 2.96],
    [4.07, 5.36, 4.66, 4.25, 4.88, 0, 5.47, 5.16],
    [4.27, 3.54, 3.77, 3.80, 1.41, 5.47, 0, 2.88],
    [5.11, 4.79, 4.90, 4.74, 2.96, 5.16, 2.88, 0]
])

hierarchical_clustering(distance_matrix, method='single')  # Change to 'average' or 'complete' for other linkages

distance_matrix = np.array([
 [0, 4, 7 , 9 , 5 , 5 , 5, 6],
 [4, 0, 7 , 7 , 7 , 3 , 7, 8],
 [7, 7, 0 , 10,  6,  6,  4, 9],
 [9, 7, 10,  0,  8,  6,  10, 9],
 [5, 7, 6 , 8 , 0 , 8 , 6, 7],
 [5, 3, 6 , 6 , 8 , 0 , 8, 11],
 [5, 7, 4 , 10,  6,  8,  0, 7],
 [6, 8, 9 , 9 , 7 , 11,  7, 0],

])
hierarchical_clustering(distance_matrix, method='average')  # Change to 'average' or 'complete' for other linkages
