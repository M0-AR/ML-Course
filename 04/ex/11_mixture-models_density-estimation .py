import numpy as np

# Given distances from O8 to O1-O7
distances = [5.11, 4.79, 4.90, 4.74, 2.96, 5.16, 2.88]

# Initialize variables
N = 7
M = 7
sigma = 1
coeff = 1/((2*np.pi)**(M/2)*np.sqrt(sigma**2))

# Calculate the density using Gaussian kernel
density_values = [coeff * np.exp(-0.5 * d**2) for d in distances]
p_O8 = sum(density_values) / N

print(p_O8)

# ----------------------- Mixture Models ----------------------
# 7 - may 2015
import math


def probability_of_class_c1(o3_o1_distance, o3_o4_distance, w1, w2):
    """
    Computes the probability that point o3 belongs to class C1.

    Parameters:
    - o3_o1_distance: Euclidean distance between point o3 and o1.
    - o3_o4_distance: Euclidean distance between point o3 and o4.
    - w1: Prior class probability for class C1.
    - w2: Prior class probability for class C2.

    Returns:
    - Probability that point o3 belongs to class C1.
    """

    numerator = w1 * math.exp(-o3_o1_distance ** 2 / 2)
    denominator = numerator + w2 * math.exp(-o3_o4_distance ** 2 / 2)

    return numerator / denominator


# Example usage:
o3_o1_distance = 4.51  # Replace with actual distance value from the table
o3_o4_distance = 3.7  # Replace with actual distance value from the table
w1 = 0.2
w2 = 0.8

prob_c1 = probability_of_class_c1(o3_o1_distance, o3_o4_distance, w1, w2)
print(f"Probability of o3 being in class C1: {prob_c1:.4f}")

