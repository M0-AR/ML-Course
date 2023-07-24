"""
02450ex_Fall2021_sol-1.pdf
Question 19
"""
import numpy as np

# Total number of observations
N = 572

# Initial weight
wi_1 = 1/N

# Accuracy of the first model
accuracy = 3/4

# Error rate of the first model
epsilon_1 = 1 - accuracy

# Weight update coefficient
alpha_1 = 0.5 * np.log((1 - epsilon_1) / epsilon_1)

# Weight update for a correctly classified observation
wi_2 = wi_1 * np.exp(-alpha_1)

# Normalization factor (sum of the updated weights for all observations)
Z = 3/4 * N * wi_1 * np.exp(-alpha_1) + 1/4 * N * wi_1 * np.exp(alpha_1)

# Normalized weight of a correctly classified observation
wi_2_normalized = wi_2 / Z

print('2/3 * 1/572 = ', 2/3 * 1/572)
print("The normalized weight of a correctly classified observation is:", wi_2_normalized)