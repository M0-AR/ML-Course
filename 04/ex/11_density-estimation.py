# 5 - may 2015
def average_relative_density(observation_index, distances, K):
    """
    Compute the average relative KNN density for a specified observation using given distances.

    Parameters:
    - observation_index (int): Index of the observation for which to compute the a.r.d.
    - distances (list of list of float): Pairwise distances between observations. `distances[i][j]` gives the distance between observation i and j.
    - K (int): Number of nearest neighbours to consider.

    Returns:
    - float: The average relative KNN density for the specified observation.
    """
    num_observations = len(distances)

    # Get the K nearest neighbours for the observation
    nearest_neighbours = sorted(range(num_observations), key=lambda x: distances[observation_index][x])[1:K + 1]

    # Compute density for the observation
    density_observation = 1 / (
            1 / K * sum([distances[observation_index][neighbour] for neighbour in nearest_neighbours]))

    # Compute densities for the nearest neighbours
    densities_neighbours = [1 / (1 / K * sum([distances[neighbour][other] for other in
                                              sorted(range(num_observations), key=lambda x: distances[neighbour][x])[
                                              1:K + 1]])) for neighbour in nearest_neighbours]

    # Compute average relative density
    ard = density_observation / (1 / K * sum(densities_neighbours))

    return ard


# Example usage
distances = [
    [0.00, 3.85, 4.51, 4.39, 4.08, 3.97, 2.18, 3.29, 5.48],
    [3.85, 0.00, 2.19, 3.46, 3.66, 3.93, 3.15, 3.47, 4.11],
    [4.51, 2.19, 0.00, 3.70, 4.30, 4.83, 3.86, 4.48, 4.19],
    [4.39, 3.46, 3.70, 0.00, 1.21, 3.09, 4.12, 3.22, 3.72],
    [4.08, 3.66, 4.30, 1.21, 0.00, 2.62, 4.30, 2.99, 4.32],
    [3.97, 3.93, 4.83, 3.09, 2.62, 0.00, 4.15, 1.29, 3.38],
    [2.18, 3.15, 3.86, 4.12, 4.30, 4.15, 0.00, 3.16, 4.33],
    [3.29, 3.47, 4.48, 3.22, 2.99, 1.29, 3.16, 0.00, 3.26],
    [5.48, 4.11, 4.19, 3.72, 4.32, 3.38, 4.33, 3.26, 0.00]
]
ard_o1 = average_relative_density(0, distances, 2)
print(f"The a.r.d. for observation o1 with K=2 is: {ard_o1:.3f}")

# 6 - may 2016
import numpy as np

# Define the distance matrix (using your provided data as an example)
distances = np.array([
    [0.00, 4.84, 0.50, 4.11, 1.07, 4.10, 4.71, 4.70, 4.93],
    [4.84, 0.00, 4.40, 5.96, 4.12, 2.01, 5.36, 3.59, 3.02],
    [0.50, 4.40, 0.00, 4.07, 0.72, 3.75, 4.66, 4.48, 4.64],
    [4.11, 5.96, 4.07, 0.00, 4.48, 4.69, 2.44, 3.68, 4.15],
    [1.07, 4.12, 0.72, 4.48, 0.00, 3.54, 4.96, 4.62, 4.71],
    [4.10, 2.01, 3.75, 4.69, 3.54, 0.00, 3.72, 2.23, 1.95],
    [4.71, 5.36, 4.66, 2.44, 4.96, 3.72, 0.00, 2.03, 2.73],
    [4.70, 3.59, 4.48, 3.68, 4.62, 2.23, 2.03, 0.00, 0.73],
    [4.93, 3.02, 4.64, 4.15, 4.71, 1.95, 2.73, 0.73, 0.00],
])

# Compute a.r.d for observation o9 with K=2
ard_o9 = average_relative_density(8, distances, 2)
print(f'a.r.d.(o9, K = 2) = {ard_o9:.4f}')

# 10 - dec 2016
distances = np.array([
    [0, 0.534, 1.257, 1.671, 1.090, 1.315, 1.484, 1.253, 1.418],
    [0.534, 0, 0.727, 2.119, 1.526, 1.689, 1.214, 0.997, 1.056],
    [1.257, 0.727, 0, 2.809, 2.220, 2.342, 1.088, 0.965, 0.807],
    [1.671, 2.119, 2.809, 0, 0.601, 0.540, 3.135, 2.908, 3.087],
    [1.090, 1.526, 2.220, 0.601, 0, 0.331, 2.563, 2.338, 2.500],
    [1.315, 1.689, 2.342, 0.540, 0.331, 0, 2.797, 2.567, 2.708],
    [1.484, 1.214, 1.088, 3.135, 2.563, 2.797, 0, 0.275, 0.298],
    [1.253, 0.997, 0.965, 2.908, 2.338, 2.567, 0.275, 0, 0.343],
    [1.418, 1.056, 0.807, 3.087, 2.500, 2.708, 0.298, 0.343, 0],
])
# Compute a.r.d for observation o4 with K=1
ard_o4 = average_relative_density(3, distances, 1)
print(f'a.r.d.(o4, K = 1) = {ard_o4:.4f}')

# 5 - dec 2018
import numpy as np


def gmm_density(distances, sigma, dimensionality):
    """
    Compute the density of a Gaussian Mixture Model for a given observation's distances.

    Parameters:
    - distances (ndarray): The distances from the observation to each mean. Shape (len(means),)
    - sigma (float): Standard deviation used for the identity covariance matrix.
    - dimensionality (int): The dimensionality of the multivariate normal distributions.

    Returns:
    - float: The computed density for the observation.
    """
    density = 0
    coeff = 1 / (np.power(2 * np.pi * sigma ** 2, dimensionality / 2))

    for d in distances:
        density += coeff * np.exp(-d ** 2 / (2 * sigma ** 2))

    return density / len(distances)

# Given distance matrix
distance_matrix = np.array([
    [0.0, 2.91, 0.63, 1.88, 1.02, 1.82, 1.92, 1.58, 1.08, 1.43],
    [2.91, 0.0, 3.23, 3.9, 2.88, 3.27, 3.48, 4.02, 3.08, 3.47],
    [0.63, 3.23, 0.0, 2.03, 1.06, 2.15, 2.11, 1.15, 1.09, 1.65],
    [1.88, 3.9, 2.03, 0.0, 2.52, 1.04, 2.25, 2.42, 2.18, 2.17],
    [1.02, 2.88, 1.06, 2.52, 0.0, 2.44, 2.38, 1.53, 1.71, 1.94],
    [1.82, 3.27, 2.15, 1.04, 2.44, 0.0, 1.93, 2.72, 1.98, 1.8],
    [1.92, 3.48, 2.11, 2.25, 2.38, 1.93, 0.0, 2.53, 2.09, 1.66],
    [1.58, 4.02, 1.15, 2.42, 1.53, 2.72, 2.53, 0.0, 1.68, 2.06],
    [1.08, 3.08, 1.09, 2.18, 1.71, 1.98, 2.09, 1.68, 0.0, 1.48],
    [1.43, 3.47, 1.65, 2.17, 1.94, 1.8, 1.66, 2.06, 1.48, 0.0]
])

# Extracting distances for o3 to o7, o8, and o9
distances_o3 = distance_matrix[2, [6, 7, 8]]
sigma = 0.5
dimensionality = 10
density_o3 = gmm_density(distances_o3, sigma, dimensionality)

print(f"Density for o3: {density_o3:.6f}")

# 24 dec 2018
def compute_posterior(probabilities, weights, component_idx):
    """
    Compute the posterior probability for an observation being assigned to a specific mixture component.

    Parameters:
    - probabilities (list[float]): Probabilities of the observation for each mixture component.
                                  Assumed to be in the order [p(xi|zi1=1), p(xi|zi2=1), ...].
    - weights (list[float]): Mixture weights for each component.
                             Assumed to be in the order [π1, π2, ...].
    - component_idx (int): Index of the mixture component for which the posterior is to be computed.

    Returns:
    - float: The computed posterior probability for the observation being assigned to the specified mixture component.
    """

    numerator = probabilities[component_idx] * weights[component_idx]
    denominator = sum(p * w for p, w in zip(probabilities, weights))

    return numerator / denominator


# Given data
probabilities = [1.25, 0.45, 0.85]
weights = [0.15, 0.53, 0.32]
# For a general K components in the Gaussian Mixture Model (GMM), the
# component would correspond to an index of k−1 in Python (or any 0-based indexing system)
component_idx = 2  # 0-indexed, so 2 corresponds to the third component

posterior = compute_posterior(probabilities, weights, component_idx)
print(f"The posterior probability γi,3 is approximately: {posterior:.2f}")

# 23 - dec 2019
import numpy as np
from scipy.stats import norm


def gmm_assignment_probability(x, cluster_idx, weights, means, variances):
    """
    Compute the probability that an observation x is assigned to a specified cluster
    in a 1D Gaussian Mixture Model.

    Parameters:
    - x (float): The observation.
    - cluster_idx (int): The 0-based index of the cluster (e.g., 1 for the second cluster).
    - weights (list of float): The mixture weights for each component.
    - means (list of float): The means for each component.
    - variances (list of float): The variances for each component.

    Returns:
    - float: The assignment probability for x to the specified cluster.
    """

    # Calculate the likelihood of x for each component
    likelihoods = [w * norm.pdf(x, loc=mu, scale=np.sqrt(sigma2))
                   for w, mu, sigma2 in zip(weights, means, variances)]

    # Compute the posterior probability using Bayes' theorem
    gamma = likelihoods[cluster_idx] / sum(likelihoods)

    return gamma


# Given parameters
weights = [0.19, 0.34, 0.48]
means = [3.177, 3.181, 3.184]
variances = [0.0062 ** 2, 0.0076 ** 2, 0.0075 ** 2]  # converting standard deviations to variances
x0 = 3.19
cluster_idx = 1  # 0-indexed, for the second cluster

probability = gmm_assignment_probability(x0, cluster_idx, weights, means, variances)
print(f"Probability that x0={x0} is assigned to cluster k={cluster_idx + 1}: {probability:.3f}")

# 25 - dec 2019
import numpy as np
from scipy.stats import norm


def kde_loo(data, sigma):
    """
    Perform leave-one-out cross-validation for Kernel Density Estimation.

    Parameters:
    - data (list of float): The observations.
    - sigma (float): The standard deviation of the Gaussian kernel.

    Returns:
    - float: Average negative log-likelihood using leave-one-out cross-validation.
    """

    N = len(data)
    loo_scores = []

    for i in range(N):
        # Leave-one-out: all data points except the i-th one
        train_data = np.delete(data, i)
        # Kernel Density Estimation for the i-th data point
        likelihoods = [norm.pdf(data[i], loc=x, scale=sigma) for x in train_data]
        p_sigma_xi = np.mean(likelihoods)
        loo_scores.append(-np.log(p_sigma_xi))

    # Average negative log-likelihood
    return np.mean(loo_scores)


# Given dataset
data = [3.918, -6.35, -2.677, -3.003]
sigma = 2

loo_error = kde_loo(data, sigma)
print(f"LOO error at sigma={sigma}: {loo_error:.3f}")

# 24 - may 2020
import numpy as np
from scipy.integrate import quad


def expected_value(pieces, bounds):
    """
    Compute the expected value for a piecewise density function.

    Parameters:
    - pieces (list of functions): List of functions that define the density in each interval.
    - bounds (list of tuple): Corresponding bounds for each function in pieces.

    Returns:
    - float: Expected value of x for the given density function.
    """

    # For each piece of the density function, compute the integral of x*p(x)
    integrals = [quad(lambda x: x * piece(x), a, b)[0] for piece, (a, b) in zip(pieces, bounds)]

    # Sum the integrals to get E[x]
    return sum(integrals)


# Define the density function p(x) in pieces
pieces = [
    lambda x: 0.6,
    lambda x: 1,
    lambda x: 1.6
]
bounds = [
    (0, 0.2),
    (0.2, 0.6),
    (0.6, 0.9)
]

E_x = expected_value(pieces, bounds)
print(f"E[x] = {E_x:.3f}")
