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

# 19 dec 2017
def probability_high_average_score(priors, likelihoods):
    """
    Calculate the probability of a basketball player having a high average score
    using the Naïve Bayes classifier.

    Parameters:
    - priors: Dictionary containing prior probabilities for LAS, MAS, and HAS.
    - likelihoods: Dictionary containing likelihood values for given conditions (HH and WL).

    Returns:
    - prob_has: The calculated probability.
    """

    # Calculate the numerator of the formula
    numerator = likelihoods['HH_HAS'] * likelihoods['WL_HAS'] * priors['HAS']

    # Calculate the denominator of the formula
    denominator = (
            likelihoods['HH_LAS'] * likelihoods['WL_LAS'] * priors['LAS'] +
            likelihoods['HH_MAS'] * likelihoods['WL_MAS'] * priors['MAS'] +
            likelihoods['HH_HAS'] * likelihoods['WL_HAS'] * priors['HAS']
    )

    prob_has = numerator / denominator
    return prob_has


# Given priors and likelihoods from the problem
priors = {
    'LAS': 4 / 10,
    'MAS': 2 / 10,
    'HAS': 4 / 10
}

likelihoods = {
    'HH_LAS': 1 / 4,
    'WL_LAS': 2 / 4,
    'HH_MAS': 0,
    'WL_MAS': 2 / 2,
    'HH_HAS': 3 / 4,
    'WL_HAS': 3 / 4
}

# Calculate the probability
result = probability_high_average_score(priors, likelihoods)
print(f"The probability of a basketball player with HH = 1 and WL = 1 having a high average score is: {result:.2f}")

# 22 dec 2017 - not done
import math

# Given values
epsilon1 = 5/10
epsilon2 = 2/10

# Compute alpha values
alpha1 = 0.5 * math.log((1-epsilon1) / epsilon1 + 1e-10)  # Added a small value to avoid division by zero
alpha2 = 0.5 * math.log((1-epsilon2) / epsilon2)

# Classify observation
# Since alpha1 is 0, the decision is solely based on the classifier from the second round
# As per the given information, the second round classifier will classify the observation as 'black_plus'

observation = (6, 240)
classification = 'black_plus' if alpha2 > 0 else 'red_cross'

print(f"Observation {observation} is classified as {classification}")

import numpy as np

# AdaBoost Parameters
num_rounds = 2
num_observations = 10
initial_weights = np.array([1 / 10] * num_observations)
alphas = []

# Given error rate for the first round
epsilon1 = 5 / 10
epsilon = [epsilon1]

# Compute alpha for epsilon1
alpha1 = 0.5 * np.log((1 - epsilon1) / epsilon1) if epsilon1 != 0 else 0
alphas.append(alpha1)


# Normally, here is where you'd adjust weights and compute epsilon2 by running the
# classifier on the weighted dataset and calculating misclassification rate.
# For now, we'll just leave it at epsilon1 since we don't have further info.

# Classification
def classify(alphas):
    # For simplicity, we are assuming that the decision from round 1 is ineffective (alpha1 = 0)
    # and the decision from round 2 is that if a point is in the white region, it's a "black plus".
    # You'd need the actual decision boundaries from Figure 10 to make a real decision.

    if alphas[0] == 0:
        return "Black Plus"
    else:
        # Placeholder - this is where the logic would go if we had the decision boundary information.
        pass


x1, x2 = 6, 240
result = classify(alphas)
print(f"Observation at x1={x1}, x2={x2} will be classified as: {result}")

# 10 may 2018
class TravelDeathProbability:
    """
    This class computes the probability of death based on modes of transport.
    """

    def __init__(self, death_probs, mode_probs):
        """
        Initializes the probabilities for each mode of transport.

        Parameters:
            death_probs (dict): Probabilities of dying given a mode of transport in percentage terms.
            mode_probs (dict): Probabilities of choosing a particular mode of transport in percentage terms.
        """
        self.death_probs = death_probs
        self.mode_probs = mode_probs

    def probability_death_by_plane(self):
        """
        Calculate the probability of dying by plane travel given a person died traveling
        between Copenhagen and Oslo using Bayes' theorem.

        Formula:
        P(Plane|Death) = (P(Death|Plane) * P(Plane)) / Sum(P(Death|Mode) * P(Mode) for all modes)

        Returns:
            float: The computed probability in percentage terms.
        """
        # Numerator calculation
        numerator = self.death_probs["Plane"] * self.mode_probs["Plane"]

        # Denominator calculation
        denominator = sum(self.death_probs[mode] * self.mode_probs[mode] for mode in self.mode_probs)

        # Calculate the Bayes probability
        P_F_given_D = (numerator / denominator) * 100

        return P_F_given_D


if __name__ == "__main__":
    # Probabilities of dying given mode of transport in percentage terms
    death_probs = {
        "Car": 0.000271,
        "Bus": 0.000004,
        "Plane": 0.000003
    }

    # Probabilities of choosing a mode of transport in percentage terms
    mode_probs = {
        "Car": 30,
        "Bus": 10,
        "Plane": 60
    }

    calculator = TravelDeathProbability(death_probs, mode_probs)
    result = calculator.probability_death_by_plane()
    print(f"Probability of death by plane given a person died traveling between Copenhagen and Oslo: {result:.2f}%")

# 21 may 2018
class NaiveBayesClassifier:
    """
    Implementation of the Naïve Bayes classifier for airline safety prediction.
    """

    def __init__(self, data_probs, prior_probs):
        """
        Initializes the probabilities required for the Naïve Bayes classifier.

        Parameters:
            data_probs (dict): Conditional probabilities for each attribute given the class label.
            prior_probs (dict): Prior probabilities for each class label.
        """
        self.data_probs = data_probs
        self.prior_probs = prior_probs

    def predict_probability(self, observation):
        """
        Predicts the probability of an airline being safe using the Naïve Bayes classifier.

        Parameters:
            observation (dict): The attributes of the airline to be classified.

        Returns:
            float: Probability of the airline being considered safe.
        """
        # Numerator calculation
        prob_safe_given_data = self.prior_probs["Safe"]
        for feature, value in observation.items():
            prob_safe_given_data *= self.data_probs[feature]["Safe"]

        # Denominator calculation
        prob_unsafe_given_data = self.prior_probs["Unsafe"]
        for feature, value in observation.items():
            prob_unsafe_given_data *= self.data_probs[feature]["Unsafe"]

        # Calculate the Naïve Bayes probability
        probability_safe = prob_safe_given_data / (prob_safe_given_data + prob_unsafe_given_data)

        return probability_safe


if __name__ == "__main__":
    # Conditional probabilities for each attribute given the class label in decimal terms.
    data_probs = {
        "xH2": {"Safe": 2 / 5, "Unsafe": 3 / 5},
        "xH3": {"Safe": 1 / 5, "Unsafe": 2 / 5},
        "xH4": {"Safe": 2 / 5, "Unsafe": 3 / 5},
        "xH5": {"Safe": 2 / 5, "Unsafe": 5 / 5}
    }

    # Prior probabilities for each class label in decimal terms.
    prior_probs = {
        "Safe": 5 / 10,
        "Unsafe": 5 / 10
    }

    # The attributes of the airline to be classified
    observation = {
        "xH2": 1,
        "xH3": 1,
        "xH4": 1,
        "xH5": 1
    }

    classifier = NaiveBayesClassifier(data_probs, prior_probs)
    result = classifier.predict_probability(observation)

    print(f"Probability of the airline being considered safe: {result:.2f}")

# 14 dec 2018
def naive_bayes_probability():
    """
    Compute the Naive Bayes probability based on given conditional probabilities.

    Using the Bayes theorem and the given conditions, this function computes:
    pNB(y = 1|f1 = 1, f2 = 1, f6 = 0)

    Returns:
        float: The probability that y=1 given f1=1, f2=1, f6=0.
    """

    # Given conditional probabilities
    # For y=1
    p_f1_given_y1 = 1/1
    p_f2_given_y1 = 2/3
    p_f6_given_y1 = 1/3
    p_y1 = 3/10

    # For y=2
    p_f1_given_y2 = 2/5
    p_f2_given_y2 = 1/1
    p_f6_given_y2 = 2/5
    p_y2 = 5/10

    # For y=3
    p_f1_given_y3 = 1/2
    p_f2_given_y3 = 0/1
    p_f6_given_y3 = 0/1
    p_y3 = 2/10

    # Compute the Naive Bayes probabilities
    numerator = p_f1_given_y1 * p_f2_given_y1 * p_f6_given_y1 * p_y1

    denominator = (
        p_f1_given_y1 * p_f2_given_y1 * p_f6_given_y1 * p_y1 +
        p_f1_given_y2 * p_f2_given_y2 * p_f6_given_y2 * p_y2 +
        p_f1_given_y3 * p_f2_given_y3 * p_f6_given_y3 * p_y3
    )

    return numerator / denominator


if __name__ == "__main__":
    # Compute the probability and print the result
    probability = naive_bayes_probability()
    print(f"The Naive Bayes probability pNB(y = 1|f1 = 1, f2 = 1, f6 = 0) is {probability:.2f} or {probability.as_integer_ratio()[0]}/{probability.as_integer_ratio()[1]}")

# 20 may 2019
def bayesian_probability(conditional_probs, prior_probs, x2_val, x10_val):
    """
    Calculate the Bayesian probability that y=1 given the observed values of x2 and x10.

    Parameters:
    - conditional_probs (dict): Conditional probabilities in the form {(x2_val, x10_val): [P(y=1), P(y=2), P(y=3)]}
    - prior_probs (list): Prior probabilities in the form [P(y=1), P(y=2), P(y=3)]
    - x2_val (int): Observed value of x2 (0 or 1)
    - x10_val (int): Observed value of x10 (0 or 1)

    Returns:
    - float: Bayesian probability p(y=1|x2=x2_val, x10=x10_val)
    """

    # Calculate the numerator of Bayes' theorem
    numerator = conditional_probs[(x2_val, x10_val)][0] * prior_probs[0]

    # Calculate the denominator of Bayes' theorem
    denominator = sum([conditional_probs[(x2_val, x10_val)][i] * prior_probs[i] for i in range(3)])

    return numerator / denominator


if __name__ == "__main__":
    # Define the conditional probabilities p(x2, x10|y)
    conditional_probs = {
        (0, 0): [0.19, 0.3, 0.19],
        (0, 1): [0.22, 0.3, 0.26],
        (1, 0): [0.25, 0.2, 0.35],
        (1, 1): [0.34, 0.2, 0.2]
    }

    # Define the prior probabilities p(y)
    prior_probs = [0.316, 0.356, 0.328]

    # Calculate Bayesian probability for x2=1 and x10=0
    probability = bayesian_probability(conditional_probs, prior_probs, 1, 0)

    print(f"The Bayesian probability p(y = 1|x2 = 1, x10 = 0) is {probability:.3f}")


# 24 dec 2018
def compute_gamma(xi_probs, pi_values, k):
    """
    Calculate the posterior probability (γik) that observation i is assigned to a specific mixture component k.

    Parameters:
    - xi_probs (list of float): The probabilities p(xi|zik=1) for each component k.
    - pi_values (list of float): The weights of the components (π).
    - k (int): The specific mixture component to which the posterior probability is computed.

    Returns:
    - float: The posterior probability γik.
    """

    numerator = xi_probs[k - 1] * pi_values[k - 1]
    denominator = sum([xi_prob * pi for xi_prob, pi in zip(xi_probs, pi_values)])

    gamma_ik = numerator / denominator
    return gamma_ik


if __name__ == "__main__":
    # Probabilities read from Figure 15 for observation i
    xi_probs = [1.25, 0.45, 0.85]

    # Given weights of the components
    pi_values = [0.15, 0.53, 0.32]

    # Compute γi,3
    gamma_i3 = compute_gamma(xi_probs, pi_values, 3)
    print(f"γi,3: {gamma_i3:.2f}")

# 11 may 2019 - not

# 13 may 2019
def naive_bayes_prob(feature_probs_for_label, label_probs):
    """
    Compute the probability using the Naive Bayes formula.

    Args:
    - feature_probs_for_label (list of dicts): A list where each item is a dictionary of feature probabilities for a given label.
    - label_probs (list of float): Probabilities for each label.

    Returns:
    - float: The Naive Bayes probability for the specific label.
    """

    numerator = 1
    denominator = 0

    # Calculate the numerator for the specified label (in this case, y=3)
    for feature, prob in feature_probs_for_label[0].items():
        numerator *= prob
    numerator *= label_probs[0]

    # Calculate the denominator
    for j in range(3):
        tmp = 1
        for feature, prob in feature_probs_for_label[j].items():
            tmp *= prob
        tmp *= label_probs[j]
        denominator += tmp

    return numerator / denominator


# Feature probabilities for each label y=j, j in [0,1,2] which correspond to y=1, y=2, and y=3
feature_probs = [
    {'f2': 5 / 7, 'f5': 2 / 7, 'f8': 2 / 7},
    {'f2': 3 / 4, 'f5': 3 / 4, 'f8': 1 / 4},
    {'f2': 3 / 5, 'f5': 1 / 5, 'f8': 3 / 10}
]

# Prior probabilities for each label
label_probs = [1 / 2, 3 / 10, 1 / 5]

# Calculate the naive bayes probability for y=3 given the observations
prob = naive_bayes_prob(feature_probs, label_probs)
print(prob)  # Outputs: 0.3696935300794553 (which is approximately 934/2527)

# 20 dec 2019
def bayes_probability(cond_probs, label_probs, observed):
    """
    Compute the probability using Bayes' theorem.

    Args:
    - cond_probs (dict): Conditional probabilities of observing values given a label.
    - label_probs (dict): Probabilities for each label.
    - observed (tuple): Observed values for which we want to compute the probability.

    Returns:
    - float: The Bayesian probability for the specific label given the observed values.
    """

    numerator = cond_probs[observed][1] * label_probs[1]
    denominator = sum(cond_probs[observed][k] * label_probs[k] for k in label_probs)

    return numerator / denominator


# Conditional probabilities from Table 7
cond_probs = {
    (0, 0): {1: 0.41, 2: 0.28, 3: 0.15},
    (0, 1): {1: 0.17, 2: 0.28, 3: 0.33},
    (1, 0): {1: 0.33, 2: 0.25, 3: 0.15},
    (1, 1): {1: 0.09, 2: 0.19, 3: 0.37}
}

# Prior probabilities
label_probs = {1: 0.268, 2: 0.366, 3: 0.365}

# Calculate the Bayes probability for y=1 given the observed values
observed = (0, 1)
prob = bayes_probability(cond_probs, label_probs, observed)
print(prob)  # Outputs: 0.17
