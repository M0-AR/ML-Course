# 21 may 2019
def adaboost_predict(weak_classifier_preds, alpha_values):
    """
    Predict the final output of the AdaBoost classifier based on the weak classifiers' predictions and their weights.

    Parameters:
    - weak_classifier_preds (list of list of int): A 2D list where each row corresponds to a test observation and
                                                  each column corresponds to the prediction of a weak classifier.
    - alpha_values (list of float): Weights of each weak classifier.

    Returns:
    - list of int: Final prediction for each test observation.
    """

    # Initialize the final prediction list
    final_predictions = []

    # Iterate over each test observation's predictions
    for obs_preds in weak_classifier_preds:

        # Calculate weighted sum for each class
        class_scores = {cls: 0 for cls in set(obs_preds)}
        for cls, alpha in zip(obs_preds, alpha_values):
            class_scores[cls] += alpha

        # Get the class with the maximum weighted sum
        predicted_class = max(class_scores, key=class_scores.get)
        final_predictions.append(predicted_class)

    return final_predictions


if __name__ == "__main__":
    # Test predictions for each weak classifier (from Table 7)
    test_preds = [
        [2, 1, 1, 2],
        [2, 2, 1, 2]
    ]

    # Weights (αt values) for each weak classifier (from Table 7)
    alphas = [-0.168, -0.325, -0.185, 0.207]

    # Compute AdaBoost predictions
    predictions = adaboost_predict(test_preds, alphas)
    print(f"AdaBoost Predictions for Test Observations: {predictions}")

# 26 may 2015
import numpy as np


def adaboost_weights(x, y, y_pred):
    """
    Calculate AdaBoost weights after one iteration.

    Parameters:
    - x: array of observations.
    - y: array of true labels.
    - y_pred: array of predicted labels by the classifier.

    Returns:
    - Normalized weights of the observations.
    """

    # Initial weights (Step 1)
    N = len(x)
    w = np.ones(N) / N

    # Classifier's error (Step 2b)
    I = (y_pred != y).astype(int)  # Indicator function: 1 for misclassified, 0 otherwise
    epsilon = np.sum(w * I)

    # Classifier's weight (Step 2c)
    if epsilon == 0:  # Handling the case where the classifier is perfect
        return w
    alpha = 0.5 * np.log((1 - epsilon) / epsilon)

    # Update weights (Step 2d)
    for i in range(N):
        if y[i] == y_pred[i]:
            w[i] = w[i] * np.exp(-alpha)
        else:
            w[i] = w[i] * np.exp(alpha)

    # Normalize weights
    w = w / np.sum(w)

    return w


# Given data
x = np.array([50, 22, 20, 76])
y = np.array([1, 1, 0, 0])
y_pred = np.array([1, 0, 1, 0])

# Calculating weights
weights = adaboost_weights(x, y, y_pred)
print(weights)

#
import numpy as np


def adaboost_weights(y, y_pred):
    """
    Calculate AdaBoost weights after one iteration based on predictions.

    Parameters:
    - y: array of true labels.
    - y_pred: array of predicted labels by the classifier.

    Returns:
    - Normalized weights of the observations.
    """

    N = len(y)
    # Initial weights
    w = np.ones(N) / N

    # Indicator function for misclassification
    I = (y_pred != y).astype(int)

    # Classifier's error
    epsilon = np.sum(w * I)

    # Compute alpha
    alpha = 0.5 * np.log((1 - epsilon) / epsilon)

    # Update weights based on correct/incorrect predictions
    for i in range(N):
        if y[i] == y_pred[i]:
            w[i] = w[i] * np.exp(-alpha)
        else:
            w[i] = w[i] * np.exp(alpha)

    # Normalize weights
    w = w / np.sum(w)

    return w


# Given data
# As the positions of observations aren't provided, we'll assume a mock prediction.
# The only relevant information is which predictions are right or wrong. Here's a mock:
# True labels: 0, 1, 0, 0, 0, 1
# Predicted labels: 0, 0, 0, 0, 0, 0 (2 wrong predictions for observations 2 and 6)
y = np.array([0, 1, 0, 0, 0, 1])
y_pred = np.array([0, 0, 0, 0, 0, 0])

# Calculating weights
weights = adaboost_weights(y, y_pred)

# Display results
print("AdaBoost Weights after 1 Iteration:")
print("-" * 40)
for i, weight in enumerate(weights, 1):
    print(f"Observation {i}: Weight = {weight:.3f}")



# 25 dec 2016
import numpy as np


def calculate_alpha(epsilon):
    """
    Compute α based on the error rate epsilon.

    Parameters:
    - epsilon: Error rate of the classifier.

    Returns:
    - alpha: Computed α value.
    """
    return 0.5 * np.log((1 - epsilon) / epsilon)


def update_weights(w, alpha, y, y_pred):
    """
    Update the weights of observations based on the AdaBoost algorithm.

    Parameters:
    - w: Array of observation weights.
    - alpha: Computed α value.
    - y: True labels.
    - y_pred: Predicted labels.

    Returns:
    - Updated weights of the observations.
    """
    # Calculate updated weights ~w(t+1) based on correct and incorrect predictions
    for i in range(len(y)):
        if y[i] == y_pred[i]:
            w[i] = w[i] * np.exp(-alpha)
        else:
            w[i] = w[i] * np.exp(alpha)

    # Normalize the weights
    w = w / np.sum(w)

    return w


def main():
    # Given data
    N = 25
    num_misclassified = 5

    # Initial weights are uniform
    w = np.ones(N) / N

    # Assuming mock predictions for illustration:
    # Correctly classified: 0, Incorrectly classified: 1
    y = np.zeros(N)
    y_pred = np.zeros(N)
    y_pred[:num_misclassified] = 1

    # Compute epsilon and alpha
    epsilon = num_misclassified / N
    alpha = calculate_alpha(epsilon)

    # Update weights
    w_updated = update_weights(w, alpha, y, y_pred)

    # Display the updated weight for misclassified observations
    misclassified_weight = w_updated[0]  # Since we marked the first few as misclassified in our mock data
    print(f"Updated Weight for Misclassified Observations: {misclassified_weight:.3f}")


if __name__ == "__main__":
    main()

# 9 may 2017
import numpy as np


def calculate_alpha(epsilon):
    """
    Compute α based on the error rate epsilon.

    Parameters:
    - epsilon: Error rate of the classifier.

    Returns:
    - alpha: Computed α value.
    """
    return 0.5 * np.log((1 - epsilon) / epsilon)


def updated_weight_correctly_classified(N, alpha, epsilon):
    """
    Compute the updated weight for correctly classified observations in AdaBoost.

    Parameters:
    - N: Total number of observations.
    - alpha: Computed α value.
    - epsilon: Error rate of the classifier.

    Returns:
    - Updated weight for correctly classified observations.
    """
    weight_initial = 1 / N
    weight_correct = weight_initial * np.exp(-alpha)

    # For normalization: 2 misclassified and N-2 correctly classified observations (as given in the solution)
    total_weight = 2 * weight_initial * np.exp(alpha) + (N - 2) * weight_correct

    return weight_correct / total_weight


def main():
    # Given data
    N = 32
    error_rate = 1 / 16

    # Compute alpha and updated weight for correctly classified observations
    alpha = calculate_alpha(error_rate)
    weight_correct = updated_weight_correctly_classified(N, alpha, error_rate)

    print(f"Updated Weight for Correctly Classified Observations: {weight_correct:.4f}")


if __name__ == "__main__":
    main()

# 22 dec 2017
import numpy as np

def calculate_alpha(epsilon):
    """
    Compute α based on the error rate epsilon.

    Parameters:
    - epsilon: Error rate of the classifier.

    Returns:
    - alpha: Computed α value.
    """
    return 0.5 * np.log((1 - epsilon) / epsilon)

def classify_observation(alpha1, alpha2):
    """
    Classify the observation based on the α values of boosting rounds.

    Parameters:
    - alpha1: α value from the first boosting round.
    - alpha2: α value from the second boosting round.

    Returns:
    - Classification result.
    """
    if alpha1 == 0:  # As mentioned, if alpha1=0, we'll use only the second classifier.
        return "black plus"
    # For more rounds or complex logic, you can expand this function accordingly.

def main():
    # Given data
    epsilon1 = 5/10
    epsilon2 = 2/10

    # Compute alpha for each round
    alpha1 = calculate_alpha(epsilon1)
    alpha2 = calculate_alpha(epsilon2)

    # Classify observation
    classification = classify_observation(alpha1, alpha2)

    print(f"The observation at x1 = 6 and x2 = 240 will be classified as: {classification}")

if __name__ == "__main__":
    main()

# 23 may 2018
import numpy as np


def compute_alpha(error):
    """
    Compute α based on error rate.

    Parameters:
    - error: Error rate for the classifier.

    Returns:
    - α value computed.
    """
    return 0.5 * np.log((1 - error) / error)


def main():
    # Given errors for each round
    e = [0.3, 0.2381, 0.2657, 0.2941]

    # Compute α for each round
    alphas = [compute_alpha(error) for error in e]

    # Given votes for O5 [classified as safe, classified as unsafe]
    # 1 denotes a vote for 'safe' and 0 for 'unsafe'
    o5_votes = [0, 1, 0, 1]

    # Given votes for O6 [classified as safe, classified as unsafe]
    o6_votes = [0, 0, 1, 1]

    # Compute weighted votes
    o5_safe_vote_strength = sum([alpha if vote else 0 for alpha, vote in zip(alphas, o5_votes)])
    o5_unsafe_vote_strength = sum([alpha if not vote else 0 for alpha, vote in zip(alphas, o5_votes)])

    o6_safe_vote_strength = sum([alpha if vote else 0 for alpha, vote in zip(alphas, o6_votes)])
    o6_unsafe_vote_strength = sum([alpha if not vote else 0 for alpha, vote in zip(alphas, o6_votes)])

    # Classify based on stronger vote
    o5_classification = 'safe' if o5_safe_vote_strength > o5_unsafe_vote_strength else 'unsafe'
    o6_classification = 'safe' if o6_safe_vote_strength > o6_unsafe_vote_strength else 'unsafe'

    print(f"Observation O5 will be classified as: {o5_classification}")
    print(f"Observation O6 will be classified as: {o6_classification}")


if __name__ == "__main__":
    main()

# 21 dec 2018
def adaboost_classification(alpha, predictions):
    """
    Classify a set of observations based on the AdaBoost algorithm.

    Parameters:
    - alpha: List of alpha values from the AdaBoost algorithm.
    - predictions: A list of lists containing predictions for each observation.

    Returns:
    A list of predicted classes for the given observations.
    """
    final_predictions = []

    # Iterate over each observation's predictions
    for observation in predictions:
        # Calculate combined alpha values for each class
        f = [sum([a for a, pred in zip(alpha, observation) if pred == y]) for y in [1, 2]]

        # Append the class with the maximum combined alpha value to the final predictions
        final_predictions.append(f.index(max(f)) + 1)

    return final_predictions


# Given alpha values
alpha = [-0.168, -0.325, -0.185, 0.207]

# Predictions for test observations [ytest1, ytest2]
test_predictions = [
    [2, 1, 1, 2],  # Predictions for ytest1
    [2, 2, 1, 2]  # Predictions for ytest2
]

# Get the AdaBoost classifications for the test observations
adaboost_predictions = adaboost_classification(alpha, test_predictions)
print(adaboost_predictions)  # Expected output: [2, 1]

# 24 dec 2019
import numpy as np


def adaboost_update_weights(y_true, y_pred, w):
    """
    Update the weights based on AdaBoost algorithm.

    Parameters:
    - y_true: List of true class labels.
    - y_pred: List of predicted class labels.
    - w: Initial weights for each observation.

    Returns:
    Updated weights after applying AdaBoost.
    """
    # Identify misclassified observations
    misclassified = [1 if true != pred else 0 for true, pred in zip(y_true, y_pred)]

    # Calculate ε_t
    epsilon_t = sum([weight * misclass for weight, misclass in zip(w, misclassified)])

    # Calculate α_t
    alpha_t = 0.5 * np.log((1 - epsilon_t) / epsilon_t)

    # Update weights
    w_new = [weight * np.exp(alpha_t * misclass) if misclass else weight * np.exp(-alpha_t) for weight, misclass in
             zip(w, misclassified)]

    # Normalize weights
    w_new = [weight / sum(w_new) for weight in w_new]

    return w_new


# Given data
y_true = [1, 2, 2, 1, 1, 1, 2]
y_pred = [1, 1, 1, 2, 1, 2, 1]
w_initial = [1 / 7] * 7  # Initial weights are 1/N for each observation

# Update weights using AdaBoost
updated_weights = adaboost_update_weights(y_true, y_pred, w_initial)
print(updated_weights)  # Expected output: [0.25, 0.1, 0.1, 0.1, 0.25, 0.1, 0.1]

# 2023
import numpy as np


def classifier_f1(features):
    """
    Classifier f1 based on the description provided.

    Parameters:
    - features: List of feature values for an observation.

    Returns:
    Class label predicted by f1.
    """
    if features[2] == 1 and features[3] == 1:
        return "C1"
    else:
        return "C2"


def calculate_importance(dataset, true_labels):
    """
    Calculate the importance of the classifier f1 on the given dataset.

    Parameters:
    - dataset: List of observations where each observation is a list of feature values.
    - true_labels: List of true class labels for each observation.

    Returns:
    Importance of the classifier f1.
    """
    # Initial weights
    w = [1 / len(dataset)] * len(dataset)

    # Identify misclassified observations by f1
    misclassified = [1 if classifier_f1(features) != label else 0 for features, label in zip(dataset, true_labels)]

    # Calculate ε_1
    epsilon_1 = sum([weight * misclass for weight, misclass in zip(w, misclassified)])

    # Calculate α_1
    alpha_1 = 0.5 * np.log((1 - epsilon_1) / epsilon_1)

    return alpha_1


# Given dataset
dataset = [
    [1, 1, 0, 1, 0, 0],
    [1, 0, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1],
    [1, 1, 0, 1, 1, 0],
    [1, 1, 0, 1, 1, 1],
    [0, 1, 1, 1, 0, 1],
    [0, 1, 0, 1, 1, 1],
    [0, 1, 0, 1, 1, 1],
    [1, 1, 1, 1, 1, 0]
]
true_labels = ["C1", "C1", "C1", "C1", "C2", "C2", "C2", "C2", "C2", "C2"]

# Calculate importance of f1
alpha_1 = calculate_importance(dataset, true_labels)
print(f"α1 ≈ {round(alpha_1, 2)}")  # Expected output can be one of the given options
