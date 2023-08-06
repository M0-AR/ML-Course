"""
Created on Sun Dec 11 16:17:14 2022

@author:

Description:
    This script computes the error term `E` using a given set of features `X`,
    target values `y`, weights `w`, and bias `w0`. The error term is a combination
    of the sum of squared differences between predictions and actual values and
    the L2 regularization term.
"""

import numpy as np
from scipy.stats import zscore


def compute_error(Xs, y, w, w0, l):
    """
    Computes the error term using the formula:
    E = ∑(y - (w0 + Xs * w))^2 + l * ∑w^2

    Args:
    - Xs (numpy.ndarray): Standardized feature matrix.
    - y (numpy.ndarray): Target values.
    - w (numpy.ndarray): Weights.
    - w0 (float): Bias term.
    - l (float): Regularization strength (lambda).

    Returns:
    - float: Computed error term.
    """
    # Compute the prediction error
    prediction_error = y - (w0 * np.ones(len(y)) + Xs @ w)

    # Calculate the error term
    E = np.sum(np.square(prediction_error)) + l * np.sum(w ** 2)

    return E


# Sample data
X = np.array([2, 5, 6, 7]).reshape(-1, 1)
y = np.array([6, 7, 7, 9])

# Standardize the feature matrix
Xs = zscore(X, ddof=1)

# Given parameters
w = np.array([0.6])
w0 = np.mean(y)
l = 2

# Compute and print the error term
E = compute_error(Xs, y, w, w0, l)
print(f"Computed Error Term: {E:.2f}")

"""
Given the result "Computed Error Term: 2.66", here's a breakdown of its significance:

Error Term:
The computed error term, E, is 2.66. This value represents the overall discrepancy between the predicted values (using the given weights, bias, and the feature matrix) and the actual target values. It also includes a penalty for the magnitude of the weights due to regularization.
Significance of the Error Value:

The error term is a critical metric in regression tasks. It indicates how well (or poorly) the model's predictions align with the actual values. A lower error term indicates that the model is making predictions closer to the real values, while a higher error term signifies larger discrepancies.
In this context, an error term of 2.66 provides a quantitative measure of the model's performance. However, to understand whether this is a "good" or "bad" value, one would need to compare it to other models or benchmarks.
Regularization's Effect:

A part of this error (2.66) comes from the regularization term. The regularization, controlled by the parameter λ (which is set to 2 in the code), penalizes the model if the weights become too large. This ensures that the model doesn't overfit the data, especially when there are many features or when the data is sparse.
Bias and Standardization:

The bias, �0 w0, is computed as the mean of the target values. This bias helps in adjusting the regression line (or hyperplane for higher dimensions) so that it better fits the data.
The features are standardized, which means they've been scaled to have a mean of zero and a standard deviation of one. This can make optimization algorithms more efficient and can also improve the interpretability of the weight values.
In summary, the error term of 2.66 provides a measure of the model's current performance, factoring in both the prediction discrepancies and the penalty due to regularization. This value can be used as feedback during the model training process, guiding updates to the weights in the pursuit of a lower error.
"""