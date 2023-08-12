# 8 - may 2016
def compute_inner_product(norm_a: float, norm_b: float, distance: float) -> float:
    """
    Compute the inner product between two vectors given their norms and the pairwise distance.

    Parameters:
    - norm_a (float): The Euclidean norm of vector a.
    - norm_b (float): The Euclidean norm of vector b.
    - distance (float): The pairwise Euclidean distance between vectors a and b.

    Returns:
    float: The inner product between vectors a and b.
    """
    return (norm_a ** 2 + norm_b ** 2 - distance ** 2) / 2


def cosine_similarity(inner_product: float, norm_a: float, norm_b: float) -> float:
    """
    Calculate the cosine similarity between two vectors given their inner product and norms.

    Parameters:
    - inner_product (float): The inner product between vectors a and b.
    - norm_a (float): The Euclidean norm of vector a.
    - norm_b (float): The Euclidean norm of vector b.

    Returns:
    float: The cosine similarity between vectors a and b.
    """
    return inner_product / (norm_a * norm_b)


# Given data
norm_o2 = 3.04
norm_o3 = 1.5
distance_o2_o3 = 4.40

# Calculate inner product and cosine similarity
inner_product = compute_inner_product(norm_o2, norm_o3, distance_o2_o3)
similarity = cosine_similarity(inner_product, norm_o2, norm_o3)

print(f"Inner Product between o2 and o3: {inner_product:.10f}")
print(f"Cosine Similarity between o2 and o3: {similarity:.10f}")

# 13 - may 2016
from typing import List

def cosine_similarity(A: List[int], B: List[int]) -> float:
    """
    Calculate the Cosine Similarity for two binary vectors.
    """
    dot_product = sum(a * b for a, b in zip(A, B))
    norm_A = sum(a ** 2 for a in A) ** 0.5
    norm_B = sum(b ** 2 for b in B) ** 0.5
    return dot_product / (norm_A * norm_B)

def jaccard_similarity(A: List[int], B: List[int]) -> float:
    """
    Calculate the Jaccard Similarity for two binary vectors.
    """
    intersection = sum(a & b for a, b in zip(A, B))
    union = sum(a | b for a, b in zip(A, B))
    return intersection / union

def smc(A: List[int], B: List[int]) -> float:
    """
    Calculate the Simple Matching Coefficient for two binary vectors.
    """
    return sum(1 for a, b in zip(A, B) if a == b) / len(A)

# Test the functions
o1 = [0, 1, 1, 0, 1]
o2 = [0, 0, 1, 0, 0]
o3 = [1, 0, 0, 0, 1]

cos_o1_o2 = cosine_similarity(o1, o2)
cos_o1_o3 = cosine_similarity(o1, o3)
j_o1_o3 = jaccard_similarity(o1, o3)
smc_o1_o2 = smc(o1, o2)

print(f"COS(o1, o2) = {cos_o1_o2:.10f}")
print(f"COS(o1, o3) = {cos_o1_o3:.10f}")
print(f"J(o1, o3) = {j_o1_o3:.10f}")
print(f"SMC(o1, o2) = {smc_o1_o2:.10f}")

# 19 - dec 2016
import numpy as np


def jaccard_similarity(r, s):
    """
    Calculate the Jaccard similarity between two binary vectors.

    Args:
    - r (list): first binary vector
    - s (list): second binary vector

    Returns:
    - float: Jaccard similarity
    """
    f11 = np.sum(np.logical_and(r, s))
    M = len(r)
    f00 = M - np.sum(np.logical_or(r, s))

    return f11 / (M - f00)


def smc_similarity(r, s):
    """
    Calculate the Simple Matching Coefficient (SMC) between two binary vectors.

    Args:
    - r (list): first binary vector
    - s (list): second binary vector

    Returns:
    - float: SMC similarity
    """
    f11 = np.sum(np.logical_and(r, s))
    f00 = np.sum(np.logical_not(np.logical_or(r, s)))
    M = len(r)

    return (f11 + f00) / M


def cosine_similarity(r, s):
    """
    Calculate the Cosine similarity between two binary vectors.

    Args:
    - r (list): first binary vector
    - s (list): second binary vector

    Returns:
    - float: Cosine similarity
    """
    f11 = np.sum(np.logical_and(r, s))
    norm_r = np.linalg.norm(r)
    norm_s = np.linalg.norm(s)

    return f11 / (norm_r * norm_s)


# Example usage:
r = [1, 1, 1, 1, 0, 1]
s = [1, 1, 0, 1, 0, 0]

print(f"Jaccard similarity: {jaccard_similarity(r, s):.4f}")
print(f"SMC similarity: {smc_similarity(r, s):.4f}")
print(f"Cosine similarity: {cosine_similarity(r, s):.4f}")

# 20 may 2017
import numpy as np


def jaccard_similarity(a, b):
    """
    Calculate the Jaccard similarity between two binary vectors.

    Args:
    - a (list): First binary vector.
    - b (list): Second binary vector.

    Returns:
    - float: Jaccard similarity.
    """
    f11 = np.sum(np.logical_and(a, b))
    M = len(a)
    f00 = M - np.sum(np.logical_or(a, b))

    return f11 / (M - f00)


def smc_similarity(a, b):
    """
    Calculate the Simple Matching Coefficient (SMC) between two binary vectors.

    Args:
    - a (list): First binary vector.
    - b (list): Second binary vector.

    Returns:
    - float: SMC similarity.
    """
    f11 = np.sum(np.logical_and(a, b))
    f00 = np.sum(np.logical_not(np.logical_or(a, b)))
    M = len(a)

    return (f11 + f00) / M


def cosine_similarity(a, b):
    """
    Calculate the Cosine similarity between two binary vectors.

    Args:
    - a (list): First binary vector.
    - b (list): Second binary vector.

    Returns:
    - float: Cosine similarity.
    """
    f11 = np.sum(np.logical_and(a, b))
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    return f11 / (norm_a * norm_b)


def p_norm_distance(a, b, p):
    """
    Calculate the p-norm distance between two vectors.

    Args:
    - a (list): First vector.
    - b (list): Second vector.
    - p (int): Norm value.

    Returns:
    - float: p-norm distance.
    """
    return np.power(np.sum(np.power(np.abs(np.subtract(a, b)), p)), 1 / p)


# Example usage:
a = [1, 0, 1, 0, 0, 1]
b = [1, 0, 1, 0, 1, 0]

print(f"Jaccard similarity: {jaccard_similarity(a, b):.4f}")
print(f"SMC similarity: {smc_similarity(a, b):.4f}")
print(f"Cosine similarity: {cosine_similarity(a, b):.4f}")
print(f"1-norm distance: {p_norm_distance(a, b, 1):.4f}")
print(f"2-norm distance: {p_norm_distance(a, b, 2):.4f}")

# 21 dec 2017
import numpy as np


def jaccard_similarity(a, b):
    """
    Calculate the Jaccard similarity between two binary vectors.

    Args:
    - a (list): First binary vector.
    - b (list): Second binary vector.

    Returns:
    - float: Jaccard similarity.
    """
    f11 = np.sum(np.logical_and(a, b))
    M = len(a)
    f00 = M - np.sum(np.logical_or(a, b))

    return f11 / (M - f00)


def k_nearest_neighbors(test_instance, training_data, labels, k=3):
    """
    Classify a test instance using the k-Nearest Neighbors algorithm.

    Args:
    - test_instance (list): The instance to classify.
    - training_data (list of lists): The training dataset.
    - labels (list): Corresponding labels for the training data.
    - k (int, optional): The number of neighbors to consider. Default is 3.

    Returns:
    - str: Predicted label for the test instance.
    """
    similarities = [jaccard_similarity(test_instance, train_instance) for train_instance in training_data]
    sorted_indices = np.argsort(similarities)[::-1]  # Sort in descending order
    k_indices = sorted_indices[:k]
    k_labels = [labels[i] for i in k_indices]

    # Return the most common label among the k neighbors
    return max(k_labels, key=k_labels.count)


# Example usage:
training_data = [
    [1, 0, 1, 0, 1, 0, 1, 0],  # O1
    # ... add vectors for O2 through O9
]

labels = [
    "blue",  # O1
    # ... add labels for O2 through O9
]

test_instance = [0, 1, 0, 1, 0, 1, 1, 0]  # O10

prediction = k_nearest_neighbors(test_instance, training_data, labels)
print(f"O10 is predicted to belong to the class: {prediction}")

# 11 may 2018
from collections import Counter
from math import sqrt


def cosine_similarity(s1, s2):
    """
    Compute the cosine similarity between two text documents using
    a bag-of-words encoding.

    Args:
    - s1 (str): The first document.
    - s2 (str): The second document.

    Returns:
    - float: Cosine similarity between s1 and s2.
    """
    # Tokenize the documents
    words1 = s1.split()
    words2 = s2.split()

    # Create word frequency dictionaries
    freq1 = Counter(words1)
    freq2 = Counter(words2)

    # Compute common words (f11)
    common_words = set(freq1.keys()) & set(freq2.keys())
    f11 = sum([freq1[word] * freq2[word] for word in common_words])

    # Compute norms of each document
    norm1 = sqrt(sum([freq1[word] ** 2 for word in freq1]))
    norm2 = sqrt(sum([freq2[word] ** 2 for word in freq2]))

    # Compute cosine similarity
    return f11 / (norm1 * norm2)


# Example usage:
s1 = "the bag of words representation should not give you a hard time"
s2 = "remember the representation should be a vector"

similarity = cosine_similarity(s1, s2)
print(f"The cosine similarity between s1 and s2 is: {similarity:.6f}")

# 17 dec 2019
import numpy as np

def cosine_similarity(vector1, vector2):
    """
    Compute cosine similarity between two binary vectors.

    Args:
    - vector1 (list or numpy array): first binary vector.
    - vector2 (list or numpy array): second binary vector.

    Returns:
    - float: cosine similarity between vector1 and vector2.
    """
    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    return dot_product / (norm1 * norm2)


def jaccard_similarity(vector1, vector2):
    """
    Compute Jaccard similarity between two binary vectors.

    Args:
    - vector1 (list or numpy array): first binary vector.
    - vector2 (list or numpy array): second binary vector.

    Returns:
    - float: Jaccard similarity between vector1 and vector2.
    """
    intersection = np.logical_and(vector1, vector2).sum()
    union = np.logical_or(vector1, vector2).sum()
    return intersection / union


def smc_similarity(vector1, vector2):
    """
    Compute Simple Matching Coefficient (SMC) between two binary vectors.

    Args:
    - vector1 (list or numpy array): first binary vector.
    - vector2 (list or numpy array): second binary vector.

    Returns:
    - float: SMC similarity between vector1 and vector2.
    """
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)
    matches = (vector1 == vector2).sum()
    total_elements = len(vector1)
    return matches / total_elements

if __name__ == "__main__":
    o1 = [0, 0, 0, 1, 0, 0, 0, 0, 0]
    o3 = [0, 1, 1, 1, 1, 1, 0, 0, 0]

    print(f"Cosine similarity between o1 and o3: {cosine_similarity(o1, o3)}")
    print(f"Jaccard similarity between o1 and o3: {jaccard_similarity(o1, o3)}")
    print(f"SMC similarity between o1 and o3: {smc_similarity(o1, o3)}")


# ----------------------------------- Cluster

# 7 dec 2019
def compute_S(Z: list, Q: list) -> int:
    """
    Compute the value S for given clusterings Z and Q.

    Parameters:
    - Z (list): Ground truth clustering.
    - Q (list): Obtained clustering.

    Returns:
    - int: The computed value of S.
    """
    N = sum([len(cluster) for cluster in Z])
    S = 0

    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            in_cluster1_same = any(i in cluster and j in cluster for cluster in Z)
            in_cluster2_same = any(i in cluster and j in cluster for cluster in Q)

            # If they are in the same cluster in both Z and Q
            if in_cluster1_same and in_cluster2_same:
                S += 1

    return S


def compute_D(Z: list, Q: list) -> int:
    """
    Compute the value D for given clusterings Z and Q.

    Parameters:
    - Z (list): Ground truth clustering.
    - Q (list): Obtained clustering.

    Returns:
    - int: The computed value of D.
    """
    N = sum([len(cluster) for cluster in Z])

    n_Z = [len(cluster) for cluster in Z]
    n_Q = [len(cluster) for cluster in Q]

    # Using the formula for D
    term1 = N * (N - 1) // 2
    term2 = sum([n * (n - 1) // 2 for n in n_Z])
    term3 = sum([n * (n - 1) // 2 for n in n_Q])
    S = compute_S(Z, Q)

    return term1 - term2 - term3 + S


def jaccard_similarity(Z: list, Q: list) -> float:
    """
    Compute the Jaccard similarity for given clusterings Z and Q.

    Parameters:
    - Z (list): Ground truth clustering.
    - Q (list): Obtained clustering.

    Returns:
    - float: The Jaccard similarity value.
    """
    N = sum([len(cluster) for cluster in Z])
    S = compute_S(Z, Q)
    D = compute_D(Z, Q)

    return S / (0.5 * N * (N - 1) - D + S)


# Given clusters
Z = [{1, 2}, {3, 4, 5}, {6, 7, 8, 9, 10}]
Q = [{10}, {1, 2, 4, 5, 6, 7}, {3, 8, 9}]

similarity = jaccard_similarity(Z, Q)
print(f"The Jaccard similarity between Z and Q is: {similarity:.3f}")

import numpy as np
from scipy.special import comb


def compute_counting_matrix(Z, Q):
    """
    Compute the counting matrix for two clusterings Z and Q.

    Parameters:
    - Z (list[set[int]]): Clustering Z.
    - Q (list[set[int]]): Clustering Q.

    Returns:
    - np.ndarray: The counting matrix n.
    """
    n = np.zeros((len(Z), len(Q)))
    for i, z in enumerate(Z):
        for j, q in enumerate(Q):
            n[i, j] = len(z.intersection(q))
    return n


def compute_S_from_counting_matrix(n):
    """
    Compute the value of S from the counting matrix.

    Parameters:
    - n (np.ndarray): Counting matrix.

    Returns:
    - int: The value of S.
    """
    return np.trace(n)


def compute_D_from_counting_matrix(n):
    """
    Compute the value of D from the counting matrix.

    Parameters:
    - n (np.ndarray): Counting matrix.

    Returns:
    - int: The value of D.
    """
    N = np.sum(n)

    n_Z = np.sum(n, axis=1)
    n_Q = np.sum(n, axis=0)

    term1 = N * (N - 1) // 2
    term2 = np.sum([nz * (nz - 1) // 2 for nz in n_Z])
    term3 = np.sum([nq * (nq - 1) // 2 for nq in n_Q])
    S = compute_S_from_counting_matrix(n)

    return term1 - term2 - term3 + S


def jaccard_similarity(counting_matrix: np.ndarray) -> float:
    """
    Compute the Jaccard similarity for two clusterings based on their counting matrix.

    The Jaccard similarity is given by:
    J[Z, Q] = S / (0.5 * N * (N - 1) - D)

    where:
    - N is the total number of observations.
    - S is the number of pairs of observations that are in the same cluster in both Z and Q.
    - D is a computed metric that takes into account pairs of observations in the same cluster in Z or Q, but not both.

    Parameters:
    - counting_matrix (np.ndarray): A matrix where the element at [i,j] indicates the number
      of observations in the i-th cluster of Z that are also in the j-th cluster of Q.

    Returns:
    - float: The computed Jaccard similarity between the clusterings represented by the counting matrix.
    """
    N = counting_matrix.sum()
    S = compute_S_from_counting_matrix(counting_matrix)
    D = compute_D_from_counting_matrix(counting_matrix)

    # Calculate Jaccard similarity
    J = S / (0.5 * N * (N - 1) - D)

    return J


# Given clusters
Z = [{1, 2}, {3, 4, 5}, {6, 7, 8, 9, 10}]
Q = [{10}, {1, 2, 4, 5, 6, 7}, {3, 8, 9}]

# Compute counting matrix
counting_matrix = compute_counting_matrix(Z, Q)
print(f"Counting Matrix:\n{counting_matrix}")

# Compute Jaccard similarity
print(f"Jaccard Similarity: {jaccard_similarity(counting_matrix)}")


# 7 dec 2018
import numpy as np

def entropy(cluster):
    """
    Calculate entropy for a cluster.

    Args:
    - cluster (list of sets): Cluster representation with each set as a group in the cluster.

    Returns:
    - float: Entropy of the cluster.
    """
    total_points = sum(len(group) for group in cluster)
    entropy_value = -sum((len(group) / total_points) * np.log(len(group) / total_points) for group in cluster)
    return entropy_value

def joint_entropy(cluster1, cluster2):
    """
    Calculate joint entropy between two clusters.

    Args:
    - cluster1 (list of sets): First cluster representation.
    - cluster2 (list of sets): Second cluster representation.

    Returns:
    - float: Joint entropy between the two clusters.
    """
    total_points = sum(len(group) for group in cluster1)
    joint_entropy_value = 0
    for group1 in cluster1:
        for group2 in cluster2:
            intersect_size = len(group1.intersection(group2))
            if intersect_size:
                prob = intersect_size / total_points
                joint_entropy_value += prob * np.log(prob)
    return -joint_entropy_value


def normalized_mutual_information(cluster1, cluster2):
    """
    Calculate the normalized mutual information between two clusters.

    Args:
    - cluster1 (list of sets): First cluster representation.
    - cluster2 (list of sets): Second cluster representation.

    Returns:
    - float: Normalized mutual information between the two clusters.
    """
    h_cluster1 = entropy(cluster1)
    h_cluster2 = entropy(cluster2)
    h_joint = joint_entropy(cluster1, cluster2)
    mi = h_cluster1 + h_cluster2 - h_joint
    nmi = mi / np.sqrt(h_cluster1 * h_cluster2)
    return nmi

# Given clusters
Z = [{1, 2, 3},{4, 5, 6, 7, 8}, {9, 10}]
Q = [{4, 6}, {1, 3, 5, 7, 8, 9, 10}, {2}]

nmi_value = normalized_mutual_information(Z, Q)
print(f"NMI[Z, Q] = {nmi_value:.3f}")

# 8 dec 2020
import numpy as np
from scipy.special import comb

def compute_S(confusion_matrix: np.ndarray) -> int:
    """
    Compute the S value from the confusion matrix.

    Parameters:
    - confusion_matrix (np.ndarray): The confusion matrix where each element represents
                                     the number of observations jointly clustered.

    Returns:
    - int: The S value.
    """
    return sum(comb(val, 2) for val in confusion_matrix.flatten() if val > 1)


def compute_D(confusion_matrix: np.ndarray) -> int:
    """
    Compute the D value from the confusion matrix.

    Parameters:
    - confusion_matrix (np.ndarray): The confusion matrix where each element represents
                                     the number of observations jointly clustered.

    Returns:
    - int: The D value.
    """
    N = confusion_matrix.sum()
    sum_rows = confusion_matrix.sum(axis=1)
    sum_cols = confusion_matrix.sum(axis=0)

    term1 = comb(N, 2)
    term2 = sum(comb(row, 2) for row in sum_rows if row > 1)
    term3 = sum(comb(col, 2) for col in sum_cols if col > 1)
    S = compute_S(confusion_matrix)

    return term1 - term2 - term3 + S


def rand_similarity(confusion_matrix: np.ndarray) -> float:
    """
    Compute the Rand similarity for two clusterings based on their confusion matrix.

    Parameters:
    - confusion_matrix (np.ndarray): The confusion matrix where each element represents
                                     the number of observations jointly clustered.

    Returns:
    - float: The computed Rand similarity.
    """
    N = confusion_matrix.sum()
    S = compute_S(confusion_matrix)
    D = compute_D(confusion_matrix)

    return (S + D) / comb(N, 2)


# Given confusion matrix
confusion_matrix = np.array([[114, 0, 32],
                             [0, 119, 0],
                             [8, 0, 60]])

similarity = rand_similarity(confusion_matrix)
print(f"The Rand similarity is: {similarity:.2f}")

# 25 dec 2021
import numpy as np
from scipy.spatial.distance import cosine

def p_norm_distance(x: np.ndarray, y: np.ndarray, p: float) -> float:
    """
    Compute the p-norm distance between two vectors.

    Parameters:
    - x (np.ndarray): First vector.
    - y (np.ndarray): Second vector.
    - p (float): The value of p for the p-norm.

    Returns:
    - float: The p-norm distance between x and y.
    """
    if p == float('inf'):
        return np.max(np.abs(x - y))
    else:
        return np.sum(np.abs(x - y) ** p) ** (1 / p)

def cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute the cosine similarity between two vectors.

    Parameters:
    - x (np.ndarray): First vector.
    - y (np.ndarray): Second vector.

    Returns:
    - float: The cosine similarity between x and y.
    """
    return 1 - cosine(x, y)

# Example vectors
x35 = np.array([-1.24, -0.26, -1.04])
x53 = np.array([-0.60, -0.86, -0.50])

# Calculations
dp_1 = p_norm_distance(x35, x53, 1)
dp_4 = p_norm_distance(x35, x53, 4)
dp_inf = p_norm_distance(x35, x53, float('inf'))
cos_similarity = cosine_similarity(x35, x53)

print(f"dp=1(x35, x53) = {dp_1:.2f}")
print(f"dp=4(x35, x53) = {dp_4:.2f}")
print(f"dp=∞(x35, x53) = {dp_inf:.2f}")
print(f"cos(x35, x53) = {cos_similarity:.2f}")


