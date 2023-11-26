import numpy as np

# We will assume a Gaussian function for the kernel density estimator (KDE)
def gaussian_kde(x, mean, variance):
    """Returns the value of the Gaussian function for the KDE."""
    return (1.0 / np.sqrt(2 * np.pi * variance)) * np.exp(-((x - mean) ** 2) / (2 * variance))

# Given lambda values and their corresponding variances (since σ^2 = λ^2)
lambdas = [0.15, 0.21, 0.49, 1.15]
variances = [l**2 for l in lambdas]

# Observations
observations = np.array([-0.82, 0.0, 2.5])

# Function to compute the log likelihood for leave-one-out cross-validation
def compute_loo_log_likelihood(observations, variance):
    N = len(observations)
    log_likelihoods = []
    for i in range(N):
        # Leave one out cross-validation: remove the i-th observation
        loo_observations = np.delete(observations, i)
        # Calculate the KDE without the i-th observation
        kde_values = np.array([gaussian_kde(observations[i], x_j, variance) for x_j in loo_observations])
        # Calculate the log likelihood for the i-th observation
        log_likelihood = np.log(np.sum(kde_values) / (N - 1))
        log_likelihoods.append(log_likelihood)
    # Average log likelihood over all observations
    return np.sum(log_likelihoods) / N

# Dictionary to store the log likelihoods for each lambda
log_likelihoods = {}

# Calculate the log likelihoods for each variance
for var in variances:
    log_likelihoods[var] = compute_loo_log_likelihood(observations, var)

print(log_likelihoods)

import numpy as np

# Component weights
w1 = 0.5
w2 = 0.49
w3 = 0.01

# Covariance matrices
Sigma1 = np.array([[1.1, 2.0], [2.0, 5.5]])
Sigma2 = np.array([[1.1, 0.0], [0.0, 5.5]])
Sigma3 = np.array([[1.5, 0.0], [0.0, 1.5]])

# Overall covariance matrix
Sigma_overall = w1 * Sigma1 + w2 * Sigma2 + w3 * Sigma3

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eigh(Sigma_overall)

# Sort the eigenvalues and eigenvectors
sorted_indices = np.argsort(eigenvalues)[::-1]
sorted_eigenvectors = eigenvectors[:, sorted_indices]

# First and second principal components
v1 = sorted_eigenvectors[:, 0]
v2 = sorted_eigenvectors[:, 1]

# print(f'{Sigma_overall} , v1{v1} , v2{v2}')

import numpy as np

# AdaBoost importance calculation

# Given that there are three observations misclassified out of 10
total_observations = 10
misclassified_observations = 3

# Initial weights are equal for all observations, so each weight is 1/total_observations
initial_weight = 1 / total_observations
print(initial_weight)

# Weighted error rate is the sum of the weights of the misclassified samples
error_rate = misclassified_observations * initial_weight
print(error_rate)

# Calculate the importance of the classifier (alpha)
alpha = 0.5 * np.log((1 - error_rate) / error_rate)
print(alpha)

import numpy as np


def classify_point(x1, x2, A_cond, A_norm, A_less_than, B_cond, B_norm, B_less_than, C_cond, C_norm, C_less_than, D_cond, D_norm, D_less_than):
    # Calculate norms based on the conditions and specified norm type
    norm_A = np.linalg.norm(np.array([x1, x2]) - A_cond, A_norm)
    norm_B = np.linalg.norm(np.array([x1, x2]) - B_cond, B_norm)
    norm_C = np.linalg.norm(np.array([x1, x2]) - C_cond, C_norm)
    norm_D = np.linalg.norm(np.array([x1, x2]) - D_cond, D_norm)

    # Node A
    if norm_A < A_less_than:
        # Node C
        if norm_C < C_less_than:
            return 'Class 1'
        else:
            return 'Class 2'
    else:
        # Node B
        if norm_B < B_less_than:
            # Node D
            if norm_D < D_less_than:
                return 'Class 2'
            else:
                return 'Class 1'
        else:
            return 'Class 1'

# Example point to classify
x1 = 2
x2 = 4

# Define the condition points and norms for each node for Option A
A_cond_A = np.array([2, 4])
A_norm_A = 1  # L2 norm
A_less_than_A = 3
B_cond_A = np.array([6, 2])
B_norm_A = 2  # L1 norm
B_less_than_A = 3
C_cond_A = np.array([2, 4])
C_norm_A = 2  # L2 norm
C_less_than_A = 2
D_cond_A = np.array([2, 6])
D_norm_A = 1  # L1 norm
D_less_than_A = 3

# Classify the point for Option A
classification_A = classify_point(x1, x2,
                                  A_cond_A, A_norm_A, A_less_than_A,
                                  B_cond_A, B_norm_A, B_less_than_A,
                                  C_cond_A, C_norm_A, C_less_than_A,
                                  D_cond_A, D_norm_A, D_less_than_A)

print(f"Classification for Option A: {classification_A}")

# Define the condition points and norms for each node for Option A
A_cond_A = np.array([6, 2])
A_norm_A = 2  # L2 norm
A_less_than_A = 3
B_cond_A = np.array([2, 6])
B_norm_A = 1  # L1 norm
B_less_than_A = 3
C_cond_A = np.array([2, 4])
C_norm_A = 1  # L2 norm
C_less_than_A = 3
D_cond_A = np.array([2, 4])
D_norm_A = 2  # L1 norm
D_less_than_A = 2

# Classify the point for Option A
classification_A = classify_point(x1, x2,
                                  A_cond_A, A_norm_A, A_less_than_A,
                                  B_cond_A, B_norm_A, B_less_than_A,
                                  C_cond_A, C_norm_A, C_less_than_A,
                                  D_cond_A, D_norm_A, D_less_than_A)

print(f"Classification for Option B: {classification_A}")

# Define the condition points and norms for each node for Option A
A_cond_A = np.array([2, 4])
A_norm_A = 1  # L2 norm
A_less_than_A = 3
B_cond_A = np.array([2, 6])
B_norm_A = 1  # L1 norm
B_less_than_A = 3
C_cond_A = np.array([6, 2])
C_norm_A = 2  # L2 norm
C_less_than_A = 3
D_cond_A = np.array([2, 4])
D_norm_A = 2  # L1 norm
D_less_than_A = 2

# Classify the point for Option A
classification_A = classify_point(x1, x2,
                                  A_cond_A, A_norm_A, A_less_than_A,
                                  B_cond_A, B_norm_A, B_less_than_A,
                                  C_cond_A, C_norm_A, C_less_than_A,
                                  D_cond_A, D_norm_A, D_less_than_A)

print(f"Classification for Option C: {classification_A}")

# Define the condition points and norms for each node for Option A
A_cond_A = np.array([2, 4])
A_norm_A = 2  # L2 norm
A_less_than_A = 2
B_cond_A = np.array([2, 6])
B_norm_A = 1  # L1 norm
B_less_than_A = 3
C_cond_A = np.array([6, 2])
C_norm_A = 2  # L2 norm
C_less_than_A = 3
D_cond_A = np.array([2, 4])
D_norm_A = 1  # L1 norm
D_less_than_A = 3

# Classify the point for Option A
classification_A = classify_point(x1, x2,
                                  A_cond_A, A_norm_A, A_less_than_A,
                                  B_cond_A, B_norm_A, B_less_than_A,
                                  C_cond_A, C_norm_A, C_less_than_A,
                                  D_cond_A, D_norm_A, D_less_than_A)

print(f"Classification for Option D: {classification_A}")

