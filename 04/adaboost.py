"""The code snippet you provided is implementing a part of the AdaBoost algorithm, which is a popular ensemble
learning technique. Here's what's going on:

Initialization of Weights (wi): Initially, each observation is assigned an equal weight of 1/N, where N is the number
of observations. This means that initially, each observation is considered equally important for classification.

Calculating Errors (e_t): The variable e_t is calculating the error of a "weak classifier" on the dataset. It is the
sum of the weights of the incorrectly classified observations. In this case, yfalse represents where the classifier
was incorrect (true where the classifier is wrong, false where it is correct), so multiplying yfalse by the weights
and summing gives the total weight of the misclassified observations.

Calculating Classifier Coefficient (a_t): The value a_t is a coefficient that tells you how much to trust this
particular weak classifier in the final strong classifier. It's calculated based on the error rate e_t. A classifier
with lower error will have a higher a_t, meaning we trust its decisions more.

Updating Weights for Correct and Incorrect Classifications: The weights of the observations are updated differently
depending on whether the observation was classified correctly or not:

If the observation was classified correctly (ytrue), the weight is multiplied by exp(-a_t).
If the observation was classified incorrectly (yfalse), the weight is multiplied by exp(a_t).
This makes the weight of the incorrectly classified observations larger and the weight of the correctly classified observations smaller.

Normalizing the Weights (updatedWeights): The weights are then normalized so that they sum to 1. This is done by
dividing all weights by the sum of the weights.

Printing New and Unique Weights: The code finally prints the new weights and the unique values among them.

Summary The idea behind these calculations is to iteratively focus more on the observations that are difficult to
classify. By giving more weight to the incorrectly classified observations, the algorithm encourages subsequent weak
classifiers to focus on those observations. Over many iterations, this "boosts" the performance of the ensemble of
weak classifiers, forming a strong classifier that performs well even on the hard-to-classify observations. """
import numpy as np


def update_weights(y_true, y_false):
    """
    Function to perform weight update in AdaBoost algorithm.

    Args:
    y_true (numpy array): A binary array representing the correct predictions.
    y_false (numpy array): A binary array representing the incorrect predictions (complement of y_true).

    Returns:
    updated_weights (numpy array): The updated weights for each observation.
    unique_weights (numpy array): The unique weights in the updated weight array.
    """

    # Number of observations
    N = np.size(y_true)

    # Initial weights for each observation
    wi = 1 / N

    # Errors of classifier: sum of weights of incorrectly classified observations
    e_t = np.sum(y_false * wi)

    # Confidence in the current classifier
    a_t = 0.5 * np.log((1 - e_t) / e_t)

    # Update weights for correctly classified observations
    new_weight_correct = y_true * wi * np.exp(-a_t)

    # Update weights for incorrectly classified observations
    new_weight_wrong = y_false * wi * np.exp(a_t)

    # Combine updated weights
    single_weight = new_weight_correct + new_weight_wrong

    # Normalize the updated weights so they sum to 1
    updated_weights = single_weight / np.sum(single_weight)

    # Find the unique weights in the updated weight array
    unique_weights = np.unique(updated_weights)

    return updated_weights, unique_weights


if __name__ == "__main__":
    # Ground truth labels
    y_true = np.array([1, 0, 1, 0, 0, 1, 1])

    # Complementary array representing incorrect predictions
    y_false = y_true == 0

    # Update the weights
    new_weights, unique_weights = update_weights(y_true, y_false)

    print("New weights:", new_weights)
    print("Unique weights:", unique_weights)
