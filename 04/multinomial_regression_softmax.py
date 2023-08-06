"""
The given code defines a function (calculate_output_probability) to compute the output probabilities for a three-class classification problem, given the input features and weights for the first two classes. The probability for the third class is determined by the probabilities of the first two classes. Here's a breakdown of the main points:

Function Input:

X: This is the feature vector for the data point that we want to classify.
w1 and w2: These are the weight vectors for the first and second classes, respectively. The weights help us determine how each feature affects the predicted probability for each class.

Function Steps:

The scores (f1) for the first two classes are computed by taking the dot product of the input features X and the weight vectors w1 and w2. Essentially, this gives us a measure of how much each feature influences the prediction for each class.
The exponential of the scores (ef1) is then calculated. This operation ensures that our scores are non-negative and emphasizes differences between the scores.
The probabilities for the three classes are then determined. The probability for each of the first two classes is given by the exponential of its score divided by the sum of the exponentials of all the scores plus 1. The probability for the third class is calculated to ensure that the sum of all probabilities equals 1.
Function Output: The function returns an array representing the output probabilities for each of the three classes.

Code Execution:

Weights for the first (w1) and second (w2) classes are defined.
The input feature vector X is also given.
We then call our calculate_output_probability function to compute the output probabilities for the given input.

Results:

The code prints out the probabilities of the three classes for the given input feature vector. These probabilities tell us how likely it is for the input to belong to each of the three classes based on the provided weights.
In essence, this code provides a mechanism to predict the class membership probabilities of an input vector based on given weights for a three-class classification problem.
"""
import numpy as np

# Q24, Fall 2019

def calculate_output_probability(X, w1, w2):
    """
    Calculates the output probability vector for three classes given the input
    features and the weights of the first two classes. The third class is calculated
    differently by considering the probabilities of the first two classes.

    Args:
        X (numpy.ndarray): The input features vector.
        w1 (numpy.ndarray): The weights of the first class.
        w2 (numpy.ndarray): The weights of the second class.

    Returns:
        numpy.ndarray: The output probability vector for the three classes.
    """
    # Calculate the scores for the first two classes
    f1 = np.array([X @ w1.T, X @ w2.T])

    # Calculate the exponential of the scores
    ef1 = np.exp(f1)

    # Calculate the probabilities for the three classes
    y = (1 / (1 + np.sum(ef1))) * np.hstack((ef1, np.array([1])))

    return y


# Define weights for the first and second classes
w1 = np.array([0.04, 1.32, -1.48])  # First class weights
w2 = np.array([-0.03, 0.7, -0.85])  # Second class weights

# Input features
X = np.array([1, -5.52, -4.69])

# Calculate the output probability vector for the three classes
output_probabilities = calculate_output_probability(X, w1, w2)

# Display the output probabilities
print("Output Probabilities:", output_probabilities)

"""
The result "Output Probabilities: [0.26005815 0.38706989 0.35287196]" shows the probabilities of the input vector X belonging to each of the three classes:

Probability for the First Class: 0.26005815 or 26.00% - This means that there's a 26.00% chance, based on the model defined by the given weights, that the input vector X belongs to the first class.

Probability for the Second Class: 0.38706989 or 38.71% - The input vector X has a 38.71% probability of belonging to the second class.

Probability for the Third Class: 0.35287196 or 35.29% - The third class has a 35.29% likelihood of being the class of the input vector X.

In summary, based on the given weights and the model, the input vector X is most likely to belong to the second class (with a probability of 38.71%), followed closely by the third class (35.29%), and then the first class (26.00%).
"""