import numpy as np
############################################################################
# Clustering
# 5 - spring 2015
def density(distance_matrix, observation_index, K):
    """
    Compute the density for a given observation using the distance matrix.

    Args:
    - distance_matrix (numpy.ndarray): Matrix containing distances between observations.
    - observation_index (int): Index of the observation for which to compute the density.
    - K (int): Number of nearest neighbors.

    Returns:
    - float: Density for the observation.
    """
    distances = distance_matrix[observation_index]
    sorted_indices = np.argsort(distances)
    k_nearest_neighbors = sorted_indices[1:K + 1]
    return 1 / np.mean(distances[k_nearest_neighbors])


def ard(distance_matrix, observation_index, K):
    """
    Compute the average relative KNN density for a given observation.

    Args:
    - distance_matrix (numpy.ndarray): Matrix containing distances between observations.
    - observation_index (int): Index of the observation for which to compute the a.r.d.
    - K (int): Number of nearest neighbors.

    Returns:
    - float: a.r.d. for the observation.
    """
    densities = np.array([density(distance_matrix, i, K) for i in range(distance_matrix.shape[0])])
    distances = distance_matrix[observation_index]
    sorted_indices = np.argsort(distances)
    k_nearest_neighbors = sorted_indices[1:K + 1]

    return densities[observation_index] / np.mean(densities[k_nearest_neighbors])


# Sample usage:

distance_matrix = np.array([
    [0.00, 3.85, 4.51, 4.39, 4.08, 3.97, 2.18, 3.29, 5.48],
    [3.85, 0.00, 2.19, 3.46, 3.66, 3.93, 3.15, 3.47, 4.11],
    [4.51, 2.19, 0.00, 3.70, 4.30, 4.83, 3.86, 4.48, 4.19],
    [4.39, 3.46, 3.70, 0.00, 1.21, 3.09, 4.12, 3.22, 3.72],
    [4.08, 3.66, 4.30, 1.21, 0.00, 2.62, 4.30, 2.99, 4.32],
    [3.97, 3.93, 4.83, 3.09, 2.62, 0.00, 4.15, 1.29, 3.38],
    [2.18, 3.15, 3.86, 4.12, 4.30, 4.15, 0.00, 3.16, 4.33],
    [3.29, 3.47, 4.48, 3.22, 2.99, 1.29, 3.16, 0.00, 3.26],
    [5.48, 4.11, 4.19, 3.72, 4.32, 3.38, 4.33, 3.26, 0.00]
])

print(ard(distance_matrix, 0, 2))

# 3 - spring 2015
def k_nearest_neighbor_classify(distance_matrix, observation_index, class_labels):
    """
    Classify an observation based on its nearest neighbor.

    Args:
    - distance_matrix (numpy.ndarray): Matrix containing distances between observations.
    - observation_index (int): Index of the observation to classify.
    - class_labels (list): List containing the class labels for each observation.

    Returns:
    - int: Predicted class label for the observation.
    """
    distances = distance_matrix[observation_index]
    distances[observation_index] = np.inf # Exclude the observation itself
    nearest_neighbor_index = np.argmin(distances)
    return class_labels[nearest_neighbor_index]

def leave_one_out_error_rate(distance_matrix, class_labels):
    """
    Compute the leave-one-out error rate using a 1-nearest neighbor classifier.

    Args:
    - distance_matrix (numpy.ndarray): Matrix containing distances between observations.
    - class_labels (list): List containing the class labels for each observation.

    Returns:
    - float: Leave-one-out error rate.
    """
    errors = 0
    for i in range(distance_matrix.shape[0]):
        predicted_label = k_nearest_neighbor_classify(distance_matrix, i, class_labels)
        if predicted_label != class_labels[i]:
            errors += 1
    return errors / distance_matrix.shape[0]

# Sample usage:
class_labels = [1, 1, 1, 2, 2, 2, 3, 3, 3]
error_rate = leave_one_out_error_rate(distance_matrix, class_labels)
print(error_rate)

# 6 - spring 2015
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram

# Convert the full distance matrix to condensed form
condensed_matrix = distance_matrix[np.triu_indices(distance_matrix.shape[0], k=1)]

# Compute hierarchical clustering
Z = linkage(condensed_matrix, 'single', optimal_ordering=True)

# Plot dendrogram
plt.figure(figsize=(10, 5))
dendrogram(Z, labels=["o1", "o2", "o3", "o4", "o5", "o6", "o7", "o8", "o9"])
plt.title("Dendrogram of the Observations Using Minimum Linkage")
plt.show()

# 7 - spring 2015
import numpy as np

# Given values
w1 = 0.2
w2 = 0.8
M = 8

# Euclidian distances from the distance matrix
d_o3_o1 = distance_matrix[2][0] # assuming indexing starts from 0 for o1
d_o3_o4 = distance_matrix[2][3] # assuming indexing starts from 0 for o4

# Compute the multivariate normal distribution values
N_o3_u1 = (1 / (2 * np.pi)**(M/2)) * np.exp(-d_o3_o1**2 / 2)
N_o3_u2 = (1 / (2 * np.pi)**(M/2)) * np.exp(-d_o3_o4**2 / 2)

# Compute the assignment probability using Bayes rule
p_z3_C1 = (N_o3_u1 * w1) / ((N_o3_u1 * w1) + (N_o3_u2 * w2))

print(p_z3_C1)

# 8 - spring 2015
# Given values
norm_o1_square = 2.99**2
norm_o2_square = 2.26**2

# Euclidian distance from the distance matrix
d_o1_o2 = distance_matrix[0][1]

# Compute the inner product o1^T o2
inner_product_o1_o2 = (norm_o1_square + norm_o2_square - d_o1_o2**2) / 2

# Compute the extended Jaccard similarity
EJ_o1_o2 = inner_product_o1_o2 / (norm_o1_square + norm_o2_square - inner_product_o1_o2)

print(EJ_o1_o2)
############################################################################

# 9
# Compute impurities
I_F = 1 - (55/187)**2 - (44/187)**2 - (88/187)**2
I_M = 1 - (75/208)**2 - (59/208)**2 - (74/208)**2
I_A = 1 - (130/395)**2 - (103/395)**2 - (162/395)**2

# Compute the impurity gain
Delta = I_A - (187/395)*I_F - (208/395)*I_M
print(Delta)

############################################################################
