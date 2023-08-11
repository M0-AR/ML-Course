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


# 20
def compute_assignment_probability(p_values, component_index):
    """
    Compute the assignment probability of an observation to a given Gaussian component.

    Parameters:
    - p_values (list): List of probabilities of the observation under each Gaussian component.
    - component_index (int): Index of the Gaussian component (0-indexed) for which to compute assignment probability.

    Returns:
    - float: The assignment probability to the specified Gaussian component.
    """

    # Calculate the numerator (probability under the specified component)
    numerator = p_values[component_index]

    # Calculate the denominator (sum of probabilities under all components)
    denominator = sum(p_values)

    # Return the assignment probability
    return numerator / denominator


# Example usage
p_values = [0.05, 0.25, 0]  # Probabilities for observation under each Gaussian component
component_index = 0  # 0-indexed component number

prob = compute_assignment_probability(p_values, component_index)
print(f"The assignment probability to component {component_index + 1} is: {prob:.2f}")

# 22 - dec 2020
import math


def normal_density(x, mean, std_dev):
    """
    Compute the density of x under a normal distribution with specified mean and standard deviation.

    Parameters:
    - x (float): The observation.
    - mean (float): Mean of the Gaussian distribution.
    - std_dev (float): Standard deviation of the Gaussian distribution.

    Returns:
    - float: The density of x under the specified Gaussian distribution.
    """
    exponent = math.exp(-((x - mean) ** 2 / (2 * std_dev ** 2)))
    return (1 / (std_dev * math.sqrt(2 * math.pi))) * exponent


def compute_assignment_probability(x, weights, means, std_devs, component_index):
    """
    Compute the posterior probability of assignment of an observation to a given Gaussian component in a GMM.

    Parameters:
    - x (float): The observation.
    - weights (list): List of mixture weights for each Gaussian component.
    - means (list): List of means for each Gaussian component.
    - std_devs (list): List of standard deviations for each Gaussian component.
    - component_index (int): Index of the Gaussian component (0-indexed) for which to compute assignment probability.

    Returns:
    - float: The posterior probability of assignment to the specified Gaussian component.
    """

    # Compute the numerator (probability under the specified component multiplied by its weight)
    numerator = normal_density(x, means[component_index], std_devs[component_index]) * weights[component_index]

    # Compute the denominator (sum of probabilities under all components multiplied by their weights)
    denominator = sum([normal_density(x, m, s) * w for m, s, w in zip(means, std_devs, weights)])

    # Return the assignment probability
    return numerator / denominator


# Example usage
weights = [0.13, 0.55, 0.32]
means = [18.347, 14.997, 18.421]
std_devs = [1.2193, 0.986, 1.1354]
x = 15.38
component_index = 1  # For k=2 (0-indexed)

prob = compute_assignment_probability(x, weights, means, std_devs, component_index)
print(f"The posterior probability of assignment to component {component_index + 1} is: {prob:.3f}")

