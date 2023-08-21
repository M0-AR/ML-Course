import numpy as np

# Singular values from matrix S
singular_values = np.array([20.62, 13.51, 13.14, 10.94, 7.6, 3.74])

# Calculate the total variance
total_variance = np.sum(singular_values ** 2)

print((13.51**2 + 13.14**2 + 10.94**2 + 7.6**2 + 3.74**2) / total_variance)

# Calculate the cumulative proportion of variance explained
cumulative_variance_explained = np.cumsum(singular_values ** 2) / total_variance

# Print the cumulative proportion of variance explained
print("Cumulative Variance Explained: ", cumulative_variance_explained)

print(5*(3*5+1))

print((4/10*4/4*3/4*1/4)/((4/10*4/4*3/4*1/4)+(6/10*5/6*5/6*3/6)))

print((0.4+0.336+0.264+0.276+0.191)/5)
print((0.394+0.276+0.185+0.194+0.163)/5)

print(((0.42+0.39-0.35+-0.45-0.15)/5)**2)
print((10.59+0.81-1.07+3.75)**1)
print(198.2464)


print((4/10)/(7/10))


# 8 may 2018
import numpy as np


def logistic_function(z):
    """
    Compute the logistic function.

    Parameters:
    - z: Input to the logistic function

    Returns:
    - Value after applying the logistic function
    """
    return 1 / (1 + np.exp(-z))


def ann_predict(x, w1_1, w1_2, w2_0, w2_1, w2_2):
    """
    Compute the prediction using the ANN model.

    Parameters:
    - x: Input vector [x5, x6]
    - w1_1, w1_2: Weight vectors for the hidden units
    - w2_0, w2_1, w2_2: Weights for the output layer

    Returns:
    - Predicted output of the network
    """
    x = np.array([1] + x)

    # Compute outputs of the hidden units
    h1_output = logistic_function(np.dot(x, w1_1))
    h2_output = logistic_function(np.dot(x, w1_2))

    # Compute the final prediction
    prediction = w2_0 + w2_1 * h1_output + w2_2 * h2_output
    return prediction


# Define the weights
w1_1 = np.array([0.0189, 0.9159, -0.4256])
w1_2 = np.array([3.7336, -0.8003, 5.0741])
w2_0 = 0.3799e-6
w2_1 = -0.3440e-6
w2_2 = 0.0429e-6

w1_1 = np.array([-0.8, -0.3, 0.2])
w1_2 = np.array([0.3, -0.3, 0.5])
w2_0 = 1
w2_1 = -0.8
w2_2 = 0.6
# Compute the outputs for the given input points
point1 = [-2, -2]
output1 = ann_predict(point1, w1_1, w1_2, w2_0, w2_1, w2_2)

point2 = [2, 2]
output2 = ann_predict(point2, w1_1, w1_2, w2_0, w2_1, w2_2)

print(f"Output for x5 = {point1[0]}, x6 = {point1[1]}: {output1:.7e}")
print(f"Output for x5 = {point2[0]}, x6 = {point2[1]}: {output2:.7e}")
print()