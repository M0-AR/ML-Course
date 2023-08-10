def calculate_roc(data, feature_index=0):
    """
    Calculate ROC metrics for a given feature.

    Parameters:
    - data: the data set
    - feature_index: index of the feature to base the calculation on (default is 0 for x1)

    Returns:
    - FPR, TPR
    """

    # Step 1: Separate the data based on the chosen feature
    feature_pos = [row[-1] for row in data if row[feature_index] == 1]
    feature_neg = [row[-1] for row in data if row[feature_index] == 0]

    # Step 2: Determine TP, FP, TN, FN for a threshold where feature = 1 is considered positive
    TP = feature_pos.count(1)
    FP = feature_pos.count(0)
    TN = feature_neg.count(0)
    FN = feature_neg.count(1)

    # Step 3: Calculate TPR and FPR
    TPR = TP / (TP + FN) if (TP + FN) != 0 else 0
    FPR = FP / (FP + TN) if (FP + TN) != 0 else 0

    return FPR, TPR


# Sample input: This is the format for the table you provided
# Each sub-list represents a row: [x1, x2, x3, x4, x5, y]
data = [
    [1, 1, 1, 1, 0, 1],
    [0, 0, 0, 0, 0, 0],
    [1, 1, 0, 1, 0, 0],
    [0, 1, 1, 0, 1, 0],
    [1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0],
    [1, 1, 0, 1, 0, 0],
    [0, 1, 1, 0, 1, 0],
    [1, 1, 1, 1, 0, 1],
    [0, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0],
    [1, 1, 0, 1, 0, 0],
    [0, 1, 1, 0, 1, 0],
    [0, 1, 1, 0, 1, 0],
]

FPR, TPR = calculate_roc(data)
print(f"Calculated FPR: {FPR}, TPR: {TPR}")

# Based on the FPR and TPR, you would then compare against the provided ROC curves
# to determine which curve best matches the feature x1 for classification.

# -----------------------------------------------------------------------------
import numpy as np


def adaboost_weights(y_true, y_pred):
    # 1. Start with equal weights for each observation.
    n = len(y_true)
    weights = np.ones(n) / n

    # 2. Calculate the error rate (ε) based on misclassifications and weights.
    misclassified = y_true != y_pred
    epsilon = np.dot(weights, misclassified)

    # 3. Calculate the update factor α.
    alpha = 0.5 * np.log((1 - epsilon) / epsilon)

    # 4. Update weights.
    for i in range(n):
        if misclassified[i]:
            weights[i] = weights[i] * np.exp(alpha)
        else:
            weights[i] = weights[i] * np.exp(-alpha)

    # 5. Normalize the weights to sum to 1.
    weights /= np.sum(weights)

    return weights


# Example usage:
y_true = np.array([1, 1, 0, 0])
y_pred = np.array([1, 0, 0, 0])

print(adaboost_weights(y_true, y_pred))
