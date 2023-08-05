"""The code snippet you provided is mainly about evaluating the similarity between two clustering results. It uses
different metrics to compare how well the clusters in C1 correspond to those in C2. Let's go through the main
concepts and functions involved:

Confusion Matrix (n): This matrix is constructed to count the intersections between clusters in C1 and C2. Each
element n[i,j] represents the number of data points that are in cluster i in C1 and in cluster j in C2. It's
essential for calculating other metrics.

S (S): It's the sum of the pairs within clusters, where pairs are considered in both C1 and C2.

D (D): It's the sum of the pairs where the clusters disagree, i.e., the items are in the same cluster in one
clustering but in different clusters in the other.

Rand Index (rand): The Rand Index computes the similarity between two clusterings by considering all pairs of samples
and counting pairs that are either in the same cluster or in different clusters in both clusterings. The Rand Index
takes a value between 0 and 1, with 1 indicating that the two clusterings are identical.

Jaccard Coefficient (jaccard): This is similar to the Rand Index but only considers pairs of samples that are in the
same cluster in the ground truth clustering. It's a measure of agreement between C1 and C2 for pairs that are in the
same cluster.

Mutual Information (MI) and Normalized Mutual Information (NMI): These metrics provide information about the mutual
dependence between the two clusterings. MI is a measure of the shared information between C1 and C2,
while NMI normalizes this value to fall between 0 and 1, with 1 indicating perfect correspondence between the two
clusterings.

Entropy: This is used in the calculation of MI and NMI. Entropy is a measure of the uncertainty or randomness of a
system. In the context of clustering, it's used to measure how disorganized the clusters in C1 and C2 are.

Summary The code snippet calculates different metrics that can be used to evaluate how similar two given clusterings
are. Each of these metrics provides a different perspective on the similarity:

Rand Index looks at all pairs of samples. Jaccard Coefficient only considers pairs in the same cluster. MI and NMI
capture the mutual dependence between the two clusterings. These metrics are widely used in clustering analysis to
evaluate the quality of a clustering algorithm or to compare different clustering algorithms on the same dataset. """

import numpy as np
from sklearn import metrics


def number_pairs(n):
    """Calculate the number of pairs from the given matrix.

    Args:
        n (numpy.ndarray): A square matrix.

    Returns:
        int: The total number of pairs.
    """
    return np.sum((n * (n - 1)) / 2)


def rand_index(S, D, N):
    """Calculate the Rand index.

    Args:
        S (int): The number of agreements.
        D (int): The number of disagreements.
        N (int): The total number of objects.

    Returns:
        float: The Rand index.
    """
    return (S + D) / (0.5 * N * (N - 1))


def jaccard_index(S, D, N):
    """Calculate the Jaccard index.

    Args:
        S (int): The number of agreements.
        D (int): The number of disagreements.
        N (int): The total number of objects.

    Returns:
        float: The Jaccard index.
    """
    return S / (0.5 * N * (N - 1) - D)


def stats_from_matrix(n):
    """Calculate various statistics from the given matrix.

    Args:
        n (numpy.ndarray): A square matrix.

    Returns:
        tuple: A tuple containing the total values, total possible pairs,
               sum of vertical and sum of horizontal.
    """
    total_values = np.sum(n)
    total_possible_pairs = (total_values * (total_values - 1)) / 2
    sum_vertical = np.sum(n, axis=1)
    sum_horizontal = np.sum(n, axis=0)
    return total_values, total_possible_pairs, sum_vertical, sum_horizontal


def entropy(n, N):
    """Calculate the entropy of the given matrix.

    Args:
        n (numpy.ndarray): A square matrix.
        N (int): The total number of objects.

    Returns:
        float: The entropy.
    """
    p = 1 / N * n
    p = p[p != 0]
    return - np.sum(p * np.log(p))


if __name__ == "__main__":
    # Clusters
    C1 = np.array([1, 1, 1, 1, 2, 2, 3, 3, 3])
    C2 = np.array([4, 4, 1, 1, 2, 2, 2, 3, 3])

    C1 = np.array([1, 1, 1, 2, 2, 2, 2, 2, 3, 3])
    C2 = np.array([2, 3, 2, 1, 2, 1, 2, 2, 2, 2])

    # Confusion matrix (counting matrix)
    n = metrics.confusion_matrix(C1, C2)
    print("Confusion matrix:")
    print(n)

    # Calculate from matrix
    N, total_pairs, nC1, nC2 = stats_from_matrix(n)

    # Calculate S (number of agreements)
    S = number_pairs(n)
    print("S:", S)

    # Calculate D (number of disagreements)
    D = total_pairs - number_pairs(nC1) - number_pairs(nC2) + S
    print("D:", D)

    # Calculate Rand index (SMC)
    rand = rand_index(S, D, N)
    print("Rand index:", rand)

    # Calculate Jaccard index
    jaccard = jaccard_index(S, D, N)
    print("Jaccard index:", jaccard)

    # Calculate Normalized Mutual Information
    HC1 = entropy(nC1, N)
    HC2 = entropy(nC2, N)
    HC1C2 = entropy(n, N)

    MI = HC1 + HC2 - HC1C2
    NMI = MI / (np.sqrt(HC1) * np.sqrt(HC2))
    print("MI:", MI)
    print("NMI:", NMI)

"""
Confusion Matrix:

The confusion matrix represents how clusters from C1 overlap with clusters from C2.
Rows represent clusters in C1, and columns represent clusters in C2.
The value in cell [i, j] represents the number of data points that are in cluster i in C1 and in cluster j in C2.
So the matrix

[[0 2 1]
 [2 3 0]
 [0 2 0]]
 
means:
    There are 0 points that are in cluster 1 in C1 and cluster 1 in C2.
    There are 2 points that are in cluster 1 in C1 and cluster 2 in C2.
    There are 1 point that is in cluster 1 in C1 and cluster 3 in C2.
    
S (6.0): The total number of pairs of points that are clustered together in both C1 and C2. It means there are 6 such pairs where both clusterings agree.

D (15.0): The total number of pairs of points that are clustered together in one clustering but not in the other. This means there are 15 pairs where the two clusterings disagree.

Rand Index (0.4667): A measure of the similarity between the two clusterings, considering all pairs of samples. A value close to 1 indicates high similarity, and a value close to 0 indicates low similarity. A value of 0.4667 suggests moderate similarity between the two clusterings.

Jaccard Index (0.2): A measure that only considers pairs in the same cluster. The value of 0.2 indicates that 20% of the pairs that are together in one clustering are also together in the other clustering.

MI (0.2744) and NMI (0.3019): These values represent the mutual dependence between the two clusterings. MI quantifies how much knowing one clustering reduces uncertainty about the other, and NMI normalizes this value. These values suggest a mild level of agreement between the two clusterings.

Summary The confusion matrix is a detailed view of how the clusters in C1 correspond to those in C2. The other 
metrics provide summary measures of the agreement between the two clusterings, with each metric emphasizing different 
aspects of the similarity. In this specific example, the Rand Index, Jaccard Index, MI, and NMI suggest that there is 
some degree of similarity between the two clusterings, but they are far from identical. """