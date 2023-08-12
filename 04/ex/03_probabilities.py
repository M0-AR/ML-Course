# 10 may 2017
def naive_bayes_probability(feature_probs_given_class, class_priors, observed_features):
    """
    Compute the probability that a given observation belongs to a specific class using the Naive Bayes classifier.

    Parameters:
    - feature_probs_given_class (dict): A dictionary containing the probabilities of features given a class.
                                       The format should be:
                                       {'class1_feature1': prob, 'class1_feature2': prob, ...}
    - class_priors (dict): A dictionary containing the prior probabilities of each class.
                            Format: {'class1': prob, 'class2': prob, ...}
    - observed_features (dict): A dictionary of observed feature indices and their observed values.
                                Format: {feature_index: observed_value, ...}

    Returns:
    - dict: A dictionary containing the computed probabilities for each class given the observed features.
    """

    results = {}
    for class_label in class_priors.keys():
        prob = class_priors[class_label]
        for feature, observed_value in observed_features.items():
            prob_key = f"{class_label}_{feature}_{observed_value}"
            prob *= feature_probs_given_class.get(prob_key, 0)
        results[class_label] = prob

    # Normalize to make sure probabilities sum up to 1
    total_prob = sum(results.values())
    for class_label in results:
        results[class_label] /= total_prob

    return results


# Example data based on the given solution
feature_probs_given_class = {
    '0_0_0': 1 / 3, '0_1_1': 1 / 3, '0_2_1': 1 / 3,
    '1_0_0': 2 / 5, '1_1_1': 2 / 5, '1_2_1': 1
}
class_priors = {0: 6 / 11, 1: 5 / 11}
observed_features = {0: 0, 1: 1, 2: 1}

# Calculate probability for class 1 given the observed features
probabilities = naive_bayes_probability(feature_probs_given_class, class_priors, observed_features)
print(f"pNB(y = 1|f1 = 0, f2 = 1, f3 = 1) = {probabilities[1]:.4f}")

# 15 may 2016
"""
To find the probability 
p(g2=1∣y=1), we sum the probabilities for all occurrences where 
g2=1 and y=1.
Given:
P(g1=0,g2=1∣y=1)
P(g1=1,g2=1∣y=1)
The desired probability is:
p(g2=1∣y=1)=P(g1=0,g2=1∣y=1)+P(g1=1,g2=1∣y=1)
"""


def probability_g2_given_y1(joint_probabilities):
    """
    Compute the probability that a room is humid (g2 = 1) given that it is occupied (y = 1).

    Parameters:
    - joint_probabilities (dict): A dictionary containing the joint probabilities of the binary attributes and occupancy.
                                  Format: {(g1_value, g2_value, y_value): probability, ...}

    Returns:
    - float: Computed probability.
    """

    prob = 0.0
    # Sum probabilities for all occurrences where g2 = 1 and y = 1
    for (g1, g2, y), joint_prob in joint_probabilities.items():
        if g2 == 1 and y == 1:
            prob += joint_prob

    return prob


# Example data based on the given values
joint_probabilities = {
    (0, 0, 0): 0.23, (0, 1, 0): 0.40, (1, 0, 0): 0.28, (1, 1, 0): 0.09,
    (0, 0, 1): 0.01, (0, 1, 1): 0.03, (1, 0, 1): 0.46, (1, 1, 1): 0.50
}

# Calculate the probability
prob = probability_g2_given_y1(joint_probabilities)
print(f"p(g2 = 1|y = 1) ≈ {prob:.3f}")

# 21 may 2016
import numpy as np


def softmax(scores):
    """
    Compute the softmax of vector x.

    Parameters:
    - scores: array_like
        Vector of class scores

    Returns:
    - probs: array_like
        Softmax probabilities of the classes
    """
    exp_scores = np.exp(scores)
    return exp_scores / np.sum(exp_scores, axis=0)


def multinomial_regression(x, weights):
    """
    Predict class probabilities using multinomial regression with softmax.

    Parameters:
    - x: array_like
        Data point for which class probabilities need to be computed
    - weights: array_like
        Weights for the classes. Shape should be (num_classes, num_features)

    Returns:
    - probs: array_like
        Probability distribution over classes
    """
    scores = np.dot(weights, x)
    return softmax(scores)


# Sample usage
weights_options = {
    'A': np.array([[-1, -1], [1, 1], [1, -1]]),
    'B': np.array([[1, -1], [-1, -1], [1, 1]]),
    'C': np.array([[1, 1], [-1, -1], [1, -1]]),
    'D': np.array([[-1, -1], [1, 1], [-1, 1]])
}

x = np.array([0, 1])  # sample point

for option, weights in weights_options.items():
    probs = multinomial_regression(x, weights)
    print(f"Option {option}: {probs}")


# 14 dec 2016
def bayes_probability(p_x1_given_y1, p_y1, p_x1_given_y0):
    """
    Compute the probability P(y=1|x1=1) using Bayes' theorem.

    Parameters:
    - p_x1_given_y1 (float): Probability P(x1=1|y=1)
    - p_y1 (float): Prior probability P(y=1)
    - p_x1_given_y0 (float): Probability P(x1=1|y=0)

    Returns:
    - float: P(y=1|x1=1)
    """
    numerator = p_x1_given_y1 * p_y1
    denominator = p_x1_given_y1 * p_y1 + p_x1_given_y0 * (1 - p_y1)

    return numerator / denominator


# Test
p_x1_given_y1 = 0.3220
p_y1 = 0.4917
p_x1_given_y0 = 0.1639

result = bayes_probability(p_x1_given_y1, p_y1, p_x1_given_y0)
print(f"Probability P(y=1|x1=1): {result:.4f}")


# 17 dev 2016
def naive_bayes_classifier(data, features):
    """
    Compute the probability P(y=1|x1, x2, ... xn) using Naive Bayes' theorem.

    Parameters:
    - data (list of lists): The dataset where each row is an observation and each column is an attribute.
      The last column is assumed to be the target variable y.
    - features (list): A list of values for the attributes x1, x2, ..., xn.

    Returns:
    - float: P(y=1|x1, x2, ... xn)
    """
    # Validate
    if len(features) > len(data[0]) - 1:
        raise ValueError("The number of features provided is greater than the attributes in the data.")

    # Compute the priors
    p_y1 = sum(row[-1] for row in data) / len(data)
    p_y0 = 1 - p_y1

    # Compute the conditional probabilities
    p_given_y1 = 1
    p_given_y0 = 1

    for i, value in enumerate(features):
        p_given_y1 *= sum(1 for row in data if row[i] == value and row[-1] == 1) / sum(
            1 for row in data if row[-1] == 1)
        p_given_y0 *= sum(1 for row in data if row[i] == value and row[-1] == 0) / sum(
            1 for row in data if row[-1] == 0)

    # Apply the Naive Bayes formula
    numerator = p_given_y1 * p_y1
    denominator = (p_given_y1 * p_y1) + (p_given_y0 * p_y0)

    if denominator == 0:
        return 0

    return numerator / denominator


# Test
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
"""
The features parameter in the function allows you to specify which specific attribute values you're interested in. For instance:
[1, 1] means x1 = 1 and x2 = 1 
[0, 1] would mean  x1 = 0 and x2 = 1
[1, 0, 1] would mean  x1 = 1, x2 = 0 and x3 = 1
"""
result = naive_bayes_classifier(data, [1, 1])
print(f"Probability P(y=1|x1=1, x2=1): {result:.4f}")

# 16 dec 2017
def conditional_probability(p_xi_given_xj, p_xj, p_xi_given_not_xj):
    """
    Compute the conditional probability P(xj|xi) using Bayes' theorem.

    Parameters:
    - p_xi_given_xj (float): Probability P(xi|xj).
    - p_xj (float): Prior probability P(xj).
    - p_xi_given_not_xj (float): Probability P(xi|not xj).

    Returns:
    - float: P(xj|xi)
    """

    # Calculate P(not xj)
    p_not_xj = 1 - p_xj

    # Apply the Bayes formula
    numerator = p_xi_given_xj * p_xj
    denominator = (p_xi_given_xj * p_xj) + (p_xi_given_not_xj * p_not_xj)

    if denominator == 0:
        return 0

    return numerator / denominator


# Test with the given data
p_x1_given_x4 = 0.6316
p_x4 = 0.5938
p_x1_given_not_x4 = 0.1538
result = conditional_probability(p_x1_given_x4, p_x4, p_x1_given_not_x4)
print(f"Probability P(x4=0|x1=8): {result:.4f}")

# 21 may 2017
# Given probabilities
P_H = 3/8
P_L = 5/8
P_hpL_given_H = 0
P_am0_given_H = 1/3
P_hpL_given_L = 4/5
P_am0_given_L = 2/5

# Calculating P(hpL = 1, am = 0)
P_hpL_am0 = (P_hpL_given_H * P_am0_given_H * P_H) + (P_hpL_given_L * P_am0_given_L * P_L)

# Calculating P(H | hpL = 1, am = 0) using Bayes' theorem
P_H_given_hpL_am0 = (P_hpL_given_H * P_am0_given_H * P_H) / P_hpL_am0

print(f"Probability P(H|hpL=1, am=0): {P_H_given_hpL_am0:.4f}")

# 14 dec 2017
"""
Let:
A be the event that a male plays in the NBA.
B be the event that a male makes a very high salary.
We want to compute:
P(A∣B)

Which represents the probability that a male making a very high salary plays in the NBA.
According to the Bayes' theorem:

P(A∣B)= P(B) / P(B∣A)×P(A)
Where:
P(B∣A) is the probability that a male playing in the NBA has a very high salary. This is given as 1 (or 100%).
P(A) is the probability that a randomly selected male in the USA plays in the NBA. This is given as 2 out of a million or 
P(B) is the probability that a randomly selected male in the USA has a very high salary. 
This consists of two components:
The males who play in the NBA and have a very high salary.
The males who do not play in the NBA but still have a very high salary. This is given as 0.2% or 0.002.
"""
P_B_given_A = 1
P_A = 2e-6
P_B_given_not_A = 0.002

P_B = P_B_given_A * P_A + P_B_given_not_A * (1 - P_A)
P_A_given_B = (P_B_given_A * P_A) / P_B

print(f"Probability P(A|B): {P_A_given_B:.8f}")



# 20 dec 2017
def naive_bayes_classifier(data, attributes):
    """
    Compute the probability of a basketball player having a high average score given attributes using Naive Bayes.

    Parameters:
    - data (list of lists): The dataset with binary attributes. The last three columns are the class labels.
    - attributes (list of ints): List of values for the given attributes.

    Returns:
    - float: Probability of having a high average score given the attributes.
    """

    # Step 1: Calculate prior probabilities
    total_players = len(data)
    high_score_players = sum(row[-3] for row in data)
    mid_score_players = sum(row[-2] for row in data)
    low_score_players = total_players - high_score_players - mid_score_players

    p_high = high_score_players / total_players
    p_mid = mid_score_players / total_players
    p_low = low_score_players / total_players

    # Step 2: Compute conditional probabilities
    p_attributes_given_high = 1.0 if high_score_players != 0 else 0
    for i, attr_value in enumerate(attributes):
        p_attributes_given_high *= sum(row[i] for row in data if row[-3] == 1) / (high_score_players or 1)

    p_attributes_given_mid = 1.0 if mid_score_players != 0 else 0
    for i, attr_value in enumerate(attributes):
        p_attributes_given_mid *= sum(row[i] for row in data if row[-2] == 1) / (mid_score_players or 1)

    p_attributes_given_low = 1.0 if low_score_players != 0 else 0
    for i, attr_value in enumerate(attributes):
        p_attributes_given_low *= sum(row[i] for row in data if row[-1] == 1) / (low_score_players or 1)

    # Step 3: Apply Bayes' formula
    numerator = p_attributes_given_high * p_high
    denominator = (p_attributes_given_high * p_high) + (p_attributes_given_mid * p_mid) + (
                p_attributes_given_low * p_low)

    if denominator == 0:
        return 0

    return numerator / denominator


# Test
data = [
    [1, 0, 1, 0, 1, 0, 1, 0],
    [1, 0, 1, 0, 1, 0, 1, 0],
    [1, 0, 1, 0, 1, 0, 1, 0],
    [1, 0, 1, 0, 1, 0, 0, 1],
    [1, 0, 1, 0, 0, 1, 0, 1],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 1, 0, 1],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 1, 0, 1, 1, 0, 1, 0],
    [0, 1, 0, 1, 0, 1, 1, 0],
]
attributes = [1, 1]  # HH=1, WL=1
result = naive_bayes_classifier(data, attributes)
print(f"Probability of having a high average score given HH=1 and WL=1: {result:.4f}")
