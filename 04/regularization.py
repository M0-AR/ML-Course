# 02450ex_Fall2021_sol-1.pdf -
"""This script first transforms the input features X using the provided transformation function. It then computes the
necessary matrices and vectors for the least squares solution. Finally, it computes the optimal weights w_star and
prints the second weight w_star_2.

Please note that due to the nature of floating-point arithmetic, the exact result may vary slightly depending on the
environment. But it should be close to 1.5 or 3/2, which is the correct answer to the original question. """
import numpy as np

# Original dataset
X = np.array([1, 2, 3, 4])
y = np.array([6, 2, 3, 4])

# Transformation function φ(x)
X_transformed = np.array([[np.cos(np.pi/2 * xi), np.sin(np.pi/2 * xi)] for xi in X])

# Compute necessary matrices and vectors
XtX = np.matmul(X_transformed.T, X_transformed)
Xty = np.matmul(X_transformed.T, y)

# Compute the inverse of XtX
XtX_inv = np.linalg.inv(XtX)

# Compute the optimal weights
w_star = np.matmul(XtX_inv, Xty)

# Print the second weight w_star_2
print("w_star_2 =", w_star[1])