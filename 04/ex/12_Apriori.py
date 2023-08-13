# 19 dec 2021
import math

def compute_exact_updated_weight(N: int, accuracy: float) -> float:
    """
    Compute the exact updated weight of a correctly classified observation after the first round of AdaBoost.

    Parameters:
    - N (int): Total number of observations in the dataset.
    - accuracy (float): Accuracy of the first model on the full dataset.

    Returns:
    - float: Updated weight after the first round of boosting.
    """

    # Initial weight of each observation
    w_initial = 1 / N

    # Compute the error rate
    epsilon = 1 - accuracy

    # Compute the classifier's weight (alpha)
    alpha = 0.5 * math.log((1 - epsilon) / epsilon)

    # Exact formula to calculate updated weight for a correctly classified observation
    new_weight = (w_initial * math.exp(-alpha)) / (
                (accuracy * w_initial * math.exp(-alpha)) + (epsilon * w_initial * math.exp(alpha)))

    # This gives the updated weight relative to the original weight. Since
    # w(1)=1/N was the original weight, the updated weight w(2) is actually:
    return new_weight * w_initial

# Given there are 572 observations and the accuracy of the first model is 3/4
N = 572
accuracy = 3 / 4

# Compute the updated weight after the first round of boosting
updated_weight_exact = compute_exact_updated_weight(N, accuracy)
print(updated_weight_exact)


# 18 dec 2018