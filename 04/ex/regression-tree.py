# 24 dec 2021

import numpy as np

# Define the data
z = np.array([1, 3, 3, 1, 2, 3, 1])

# Compute the empirical mean
mu = np.mean(z)

# Compute the empirical variance of the data (not the mean)
variance = np.var(z, ddof=1)

# Compute the empirical variance of the mean
variance_of_mean = variance / len(z)

# Compute the standard deviation (σ) of the mean
sigma = np.sqrt(variance_of_mean)

# Compute degrees of freedom
nu = len(z) - 1

print(f"ν = {nu}")
print(f"μ = {mu:.2f}")
print(f"σ = {sigma:.2f}")
