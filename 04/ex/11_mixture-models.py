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

# 12 - dec 2017
import math


def gaussian_density(x, mu, sigma2):
    """
    Calculate Gaussian density function for a given x, mean (mu), and variance (sigma2).

    Parameters:
    - x (float): the point for which the density is being calculated
    - mu (float): the mean of the Gaussian distribution
    - sigma2 (float): the variance of the Gaussian distribution

    Returns:
    - float: the Gaussian density at point x
    """
    coefficient = 1 / math.sqrt(2 * math.pi * sigma2)
    exponential = math.exp(- (x - mu) ** 2 / (2 * sigma2))
    return coefficient * exponential


def gmm_cluster_probability(x, cluster_params):
    """
    Calculate the probability of belonging to a cluster based on the Gaussian Mixture Model.

    Parameters:
    - x (float): the point for which the cluster probability is being calculated
    - cluster_params (list of tuple): each tuple contains (weight, mean, variance) for each cluster

    Returns:
    - list of float: the probability of x belonging to each cluster
    """
    # Calculate probability for each cluster
    probabilities = [w * gaussian_density(x, mu, sigma2) for w, mu, sigma2 in cluster_params]

    # Normalize the probabilities so they sum up to 1
    total_prob = sum(probabilities)
    normalized_probs = [p / total_prob for p in probabilities]

    return normalized_probs


# Parameters for each cluster
cluster_params = [
    (0.37, 6.12, 0.09),
    (0.29, 6.55, 0.13),
    (0.34, 6.93, 0.12)
]

# Calculate the probability that observation O8 (x=6.9) is assigned to cluster 2
probabilities = gmm_cluster_probability(6.9, cluster_params)

print(f"Probability that observation O8 is assigned to cluster 2: {probabilities[1]:.2f}")

#
import numpy as np

class NaiveBayes:
    """
    A Naïve Bayes classifier for continuous features modeled using a Gaussian distribution.

    Attributes:
    - class_probs (dict): A dictionary containing the prior probabilities for each class.
    - feature_params (dict): A dictionary containing the mean and variance for each feature per class.
    - eval_count (int): Counter for the number of evaluations of the normal density function.
    """

    def __init__(self):
        self.class_probs = {}
        self.feature_params = {}
        self.eval_count = 0

    def normal_density(self, x, mu, sigma2):
        """
        Compute the Gaussian probability density for a given x, mean, and variance.

        Args:
        - x (float): The feature value.
        - mu (float): The mean of the Gaussian distribution.
        - sigma2 (float): The variance of the Gaussian distribution.

        Returns:
        - float: The Gaussian probability density.
        """
        self.eval_count += 1
        return (1.0 / np.sqrt(2 * np.pi * sigma2)) * np.exp(-((x - mu) ** 2) / (2 * sigma2))

    def predict_proba(self, x, target_class):
        """
        Compute the posterior probability for a given class and observation using the Naïve Bayes formula.

        Args:
        - x (list): A list containing the feature values for a single observation.
        - target_class (str): The class for which the posterior probability is to be computed.

        Returns:
        - float: The posterior probability of the target class for the given observation.
        """
        numerator = self.class_probs[target_class]
        for k, val in enumerate(x):
            numerator *= self.normal_density(val, self.feature_params[target_class][k]['mu'], self.feature_params[target_class][k]['sigma2'])

        denominator = sum(
            self.class_probs[c] * np.prod([
                self.normal_density(val, self.feature_params[c][k]['mu'], self.feature_params[c][k]['sigma2'])
                for k, val in enumerate(x)
            ]) for c in self.class_probs.keys()
        )

        return numerator / denominator

# Usage example
nb = NaiveBayes()

# Define the class prior probabilities and feature parameters (mean and variance) for the sake of the example
nb.class_probs = {'low_demand': 0.3, 'medium_demand': 0.3, 'high_demand': 0.4}
nb.feature_params = {
    'low_demand': [{'mu': 5, 'sigma2': 2} for _ in range(8)],
    'medium_demand': [{'mu': 6, 'sigma2': 2} for _ in range(8)],
    'high_demand': [{'mu': 7, 'sigma2': 2} for _ in range(8)]
}

# Compute the posterior probability for an observation and the "low_demand" class
observation = [5.5] * 8
posterior_proba = nb.predict_proba(observation, 'low_demand')
print(f"Posterior probability: {posterior_proba:.2f}")
print(f"Number of evaluations: {nb.eval_count}")
"""
Given the number of classes C=3 and the number of features 
M=8, the total number of evaluations is C×M=24. However, in the Naïve-Bayes formula provided, there's an extra evaluation for each feature for the target class in the numerator, which leads to an additional 
M=8 evaluations. Hence, the total evaluations become 
24+8=32, which matches with the result.
This aligns with the solution for the question which mentioned that the correct answer regarding the number of evaluations of the normal density function is 32 (option D).
"""

# 27 - may 2021
import numpy as np


def estimate_num_observations(density_at_peak: float, kernel_width: float) -> int:
    """
    Estimate the number of observations in a dataset given a KDE's density at
    an observation point and the kernel width.

    Parameters:
    - density_at_peak (float): Density value from the KDE at the observation point.
    - kernel_width (float): Width of the Gaussian kernel used in KDE. Also known as σ.

    Returns:
    - int: Estimated number of observations in the dataset.
    """

    # Compute the number of observations
    N = 1 / (np.sqrt(2 * np.pi * kernel_width ** 2) * density_at_peak)

    # Round N to the nearest integer
    return int(round(N))


# Example usage:
density_value = 0.26
kernel_width = 0.168

estimated_observations = estimate_num_observations(density_value, kernel_width)
print(f"Estimated number of observations: {estimated_observations}")

# 9 - dec 2021
import numpy as np


def kde_density(obs_distances: list, kernel_width: float, feature_dim: int) -> float:
    """
    Calculate the Kernel Density Estimation (KDE) for an observation based on distances
    to the nearest points in the dataset and the kernel width.

    Parameters:
    - obs_distances (list): List of squared distances from the observation to the nearest points.
    - kernel_width (float): Width of the Gaussian kernel used in KDE, denoted as λ.
    - feature_dim (int): Dimension of the feature space, denoted as k or M.

    Returns:
    - float: Estimated density for the observation.
    """

    # Constant factor
    const_factor = 1 / (np.sqrt(2 * np.pi * kernel_width ** 2) ** feature_dim)

    # Summation over the Gaussian evaluations for each distance
    total_sum = sum([np.exp(-d / (2 * kernel_width ** 2)) for d in obs_distances])

    # Combine the constant factor and the summation to get the KDE
    density = const_factor * total_sum / len(obs_distances)

    return density


# Example usage:
distances_to_o11 = [24.22, 39.42]
kernel_width = 20
feature_dimension = 8

estimated_density = kde_density(distances_to_o11, kernel_width, feature_dimension)
print(f"Estimated KDE density for o11: {estimated_density}")

import numpy as np

# Given values
kernel_width = 20
feature_dimension = 8

# 1. Constant factor calculation
constant_factor = 1 / np.sqrt((2 * np.pi * kernel_width**2)**feature_dimension)
print(f"Constant factor: {constant_factor}")

# 2. Exponential summation for distance 24.22
exp_sum_1 = np.exp(-24.22 / (2 * kernel_width**2))
print(f"Exponential sum for distance 24.22: {exp_sum_1}")

# 3. Exponential summation for distance 39.42
exp_sum_2 = np.exp(-39.42 / (2 * kernel_width**2))
print(f"Exponential sum for distance 39.42: {exp_sum_2}")

# Total density calculation
density = 0.5 * constant_factor * (exp_sum_1 + exp_sum_2)
print(f"Total density: {density}")

# 15 - dec 2021
import numpy as np

def gaussian_probability(x, mean, variance):
    """
    Calculate probability using Gaussian distribution formula.

    Parameters:
    - x: Value for which probability is calculated.
    - mean: Mean of the distribution.
    - variance: Variance of the distribution.

    Returns:
    - Probability value.
    """
    coeff = 1.0 / np.sqrt(2 * np.pi * variance)
    exponent = np.exp(-(x - mean) ** 2 / (2 * variance))
    return coeff * exponent

def naive_bayes_prob(x_values, mean_values, variance, class_prob, joint_prob):
    """
    Calculate the probability using the Naïve Bayes formula.

    Parameters:
    - x_values: List of attribute values.
    - mean_values: List of mean values for the class.
    - variance: Variance value.
    - class_prob: Probability of the class.
    - joint_prob: Joint probability of x_values.

    Returns:
    - Probability value based on Naïve Bayes.
    """
    probs = [gaussian_probability(x, mean, variance) for x, mean in zip(x_values, mean_values)]
    result = np.prod(probs) * class_prob
    return result / joint_prob

# Given values
x_values = [32.0, 14.0]
"""
So, the mean value for the feature x1 for class 
C1 is calculated by taking the average of two values, 38.0 and 26.8, from the table. The result is 32.4.
Similarly, the mean value for the feature x2 for class 
C1 is calculated by taking the average of two values, 15.1 and 12.8, from the table. The result is 13.95.
"""
mean_values = [32.4, 13.95]
variance = 400
class_prob = 2/11
joint_prob = 0.00010141

# Calculate probability
prob = naive_bayes_prob(x_values, mean_values, variance, class_prob, joint_prob)
print(f"pNB(C1|x1 = 32.0, x2 = 14.0) = {prob:.2%}")

