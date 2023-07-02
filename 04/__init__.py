import numpy as np
import scipy.stats as stats

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
