"""The code is a visualization and analysis of the variance explained by the principal components of a dataset. By
looking at the plot, one can determine how many components are needed to capture a significant portion of the
variance in the data, such as 80% as indicated by the threshold in this example. This information is vital in
dimensionality reduction, where you might want to represent the data using a smaller number of features that capture
most of the underlying variability.

plot_variance_explained: This function plots the variance explained by principal components, individual and
cumulative, along with a threshold line.
explained_variance_part: This function calculates and prints the cumulative
explained variance for a specific range of principal components.
Note: The indices in explained_variance_part are
inclusive, so for principal components 2 to 5 (inclusive), you would call explained_variance_part(rho, 1, 4).
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd


def plot_variance_explained(S, threshold):
    """
    Plots the variance explained by principal components.

    Parameters:
        S (array): Singular values.
        threshold (float): Threshold value for cumulative variance explained.
    """

    # Calculate the ratio of variance explained by each principal component
    rho = (S ** 2) / (S ** 2).sum()

    # Calculate cumulative sum of variance explained
    rhosum = np.cumsum(rho)

    # Plot individual and cumulative variance explained, along with threshold
    plt.figure()
    plt.plot(range(1, len(rho) + 1), rho, 'x-')
    plt.plot(range(1, len(rho) + 1), rhosum, 'o-')
    plt.plot([1, len(rho)], [threshold, threshold], 'k--')
    plt.title('Variance explained by principal components')
    plt.xlabel('Principal component')
    plt.ylabel('Variance explained')
    plt.legend(['Individual', 'Cumulative', 'Threshold'])
    plt.grid()
    plt.show()

    print("Explained variance: ", rhosum)

    return rhosum


def explained_variance_part(rho, start, end):
    """
    Calculate explained variance for a specific range of principal components.

    Parameters:
        rho (array): Ratio of variance explained by each principal component.
        start (int): Start index of the range (inclusive).
        end (int): End index of the range (inclusive).

    Returns:
        array: Cumulative explained variance for the specific range.
    """

    rhosumpart = np.cumsum(rho[start:end + 1])
    print("Explained variance from parts: ", rhosumpart)
    return rhosumpart


if __name__ == "__main__":
    S = np.array([43.4, 23.39, 18.26, 9.34, 2.14])
    threshold = 0.8

    rho = (S ** 2) / (S ** 2).sum()
    rhosum = plot_variance_explained(S, threshold)
    explained_variance_part(rho, 1, 4)  # Example for principal components 2 to 5 (inclusive)
