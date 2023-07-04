# 24
import math

# Given binarized dataset
dataset = {
    'o1': [1, 1, 0, 1, 0, 0, 'C1'],
    'o2': [1, 0, 1, 1, 0, 0, 'C1'],
    'o3': [0, 0, 1, 1, 1, 1, 'C1'],
    'o4': [0, 1, 1, 1, 1, 1, 'C1'],
    'o5': [1, 1, 0, 1, 1, 0, 'C2'],
    'o6': [1, 1, 0, 1, 1, 1, 'C2'],
    'o7': [0, 1, 1, 1, 0, 1, 'C2'],
    'o8': [0, 1, 0, 1, 1, 1, 'C2'],
    'o9': [0, 1, 0, 1, 1, 1, 'C2'],
    'o10': [1, 1, 1, 1, 1, 0, 'C2']
}

# Function representing the classifier f1
def f1(b1, b2, b3, b4, b5, b6):
    return 'C1' if b3 == 1 and b4 == 1 else 'C2'

# Calculate the weighted error ε1
num_samples = len(dataset)
sample_weight = 1 / num_samples
incorrect_classifications = 0

for observation, features in dataset.items():
    # Using f1 to classify the observation
    classification = f1(*features[:6])
    # Checking if the classification is incorrect
    if classification != features[6]:
        incorrect_classifications += 1

# Weighted error
epsilon_1 = incorrect_classifications * sample_weight

# Calculate the importance α1
alpha_1 = 0.5 * math.log((1 - epsilon_1) / epsilon_1)

# Output the result
print(f"The importance of the classifier f1, α1, is approximately {alpha_1:.2f}")


import scipy.stats as stats
# 25
import numpy as np

# Given component weights and covariance matrices
w = [0.5, 0.49, 0.01]
cov_matrices = [
    [[1.1, 2.0], [2.0, 5.5]],
    [[1.1, 0.0], [0.0, 5.5]],
    [[1.5, 0.0], [0.0, 1.5]]
]

# Sample component means for illustration (assuming non-zero means)
means = [[1.0, 1.0], [0.5, 0.5], [0.2, 0.2]]

# Compute the overall mean of the GMM
overall_mean = sum(w[k] * np.asarray(means[k]) for k in range(3))

# Compute the covariance matrix of the GMM
cov_gmm = sum(w[k] * (np.asarray(cov_matrices[k]) + np.outer(means[k], means[k])) for k in range(3)) - np.outer(overall_mean, overall_mean)

# Compute the eigenvectors of the covariance matrix
eigenvalues, eigenvectors = np.linalg.eig(cov_gmm)

# The principal component directions are the eigenvectors
# Sort them by eigenvalues
sorted_indices = np.argsort(eigenvalues)[::-1]
v1 = eigenvectors[:, sorted_indices[0]]
v2 = eigenvectors[:, sorted_indices[1]]

print("v1:", v1)
print("v2:", v2)


# 27
# Given data
X = np.array([-0.82, 0.0, 2.5])
N = len(X)
desired_L = np.array([-2.3, -2.3, -13.91])
lambda_values = [1.15, 0.15, 0.21, 0.49]

# Function to compute test log likelihood for a given λ
def compute_test_log_likelihood(X, N, lambda_value):
    L = np.zeros(N)
    for i in range(N):
        sum_gaussian_densities = 0
        for j in range(N):
            if j != i:
                # Using the Gaussian probability density function
                sum_gaussian_densities += stats.norm.pdf(X[i], loc=X[j], scale=lambda_value)
        # Compute the log density
        L[i] = np.log(sum_gaussian_densities / (N - 1))
    return L

# Check which λ gives a test log likelihood close to the desired values
for lambda_value in lambda_values:
    L = compute_test_log_likelihood(X, N, lambda_value)
    print(f"λ = {lambda_value}, L = {L}")

# Note: Due to numerical precision, you might need to check if the computed L values
# are approximately equal to the desired values.
