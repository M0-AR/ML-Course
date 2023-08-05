"""
smc_similarity function calculates the SMC similarity between two binary vectors by counting the number of matching elements and dividing by the total length of the vectors.
jaccard_similarity function calculates the Jaccard similarity between two binary vectors by finding the intersection and union of the nonzero elements.
cosine_similarity_standard function calculates the Cosine similarity between two vectors using the scikit-learn library's cosine_similarity function.
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def smc_similarity(x, y):
    """
    Calculates the Simple Matching Coefficient (SMC) similarity between two binary vectors.

    Parameters:
        x, y (array): Binary vectors to compare.

    Returns:
        float: SMC similarity between the vectors.
    """
    return np.sum(x == y) / len(x)


def jaccard_similarity(x, y):
    """
    Calculates the Jaccard similarity between two binary vectors.

    Parameters:
        x, y (array): Binary vectors to compare.

    Returns:
        float: Jaccard similarity between the vectors.
    """
    intersection = np.sum(np.logical_and(x, y))
    union = np.sum(np.logical_or(x, y))
    return intersection / union if union != 0 else 0


def cosine_similarity_standard(x, y):
    """
    Calculates the Cosine similarity between two vectors.

    Parameters:
        x, y (array): Vectors to compare.

    Returns:
        float: Cosine similarity between the vectors.
    """
    return cosine_similarity(x.reshape(1, -1), y.reshape(1, -1))[0][0]


if __name__ == "__main__":
    x = np.array([1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1])
    y = np.array([1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0])

    smc_sim = smc_similarity(x, y)
    print(f"SMC similarity: {smc_sim}")

    jaccard_sim = jaccard_similarity(x, y)
    print(f"Jaccard similarity: {jaccard_sim}")

    cosine_sim = cosine_similarity_standard(x, y)
    print(f"Cosine similarity: {cosine_sim}")
