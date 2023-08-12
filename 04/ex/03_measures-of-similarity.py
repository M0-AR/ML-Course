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
