def thresholded_linear_function(x):
    return x if x > 0 else 0


def neural_network_output(x1, x2, weights):
    # Unpack weights for clarity
    w31, w41, w32, w42, w53, w54 = weights

    # Compute Hidden Layer Outputs
    n3_input = x1 * w31 + x2 * w32
    n3_output = thresholded_linear_function(n3_input)

    n4_input = x1 * w41 + x2 * w42
    n4_output = thresholded_linear_function(n4_input)

    # Compute Output Layer
    n5_input = n3_output * w53 + n4_output * w54
    n5_output = thresholded_linear_function(n5_input)

    return n5_output


# Given weights
weights = [0.5, 0.4, -0.4, 0, -0.4, 0.1]

# Dynamic input
x1 = 1
x2 = 2

output = neural_network_output(x1, x2, weights)
print(f"The output for x1 = {x1}, x2 = {x2} is: {output:.2f}")

# 27 dec 2015
import numpy as np

def gH(t):
    """
    Logistic function as non-linearity for hidden layer neurons.

    Parameters:
    - t: input to the function

    Returns:
    - output of the logistic function
    """
    return 1 / (1 + np.exp(-t))

def gO(t):
    """
    Linear function for output layer neuron.

    Parameters:
    - t: input to the function

    Returns:
    - output of the linear function
    """
    return t

def propagate_through_network(weights, x1, x2):
    """
    Propagates input through the artificial neural network.

    Parameters:
    - weights: a dictionary containing weights for the network connections.
        For example:
        {
            "w1": weight from x1 to n3,
            "w2": weight from x2 to n3,
            ...
        }
    - x1: input to the first neuron
    - x2: input to the second neuron

    Returns:
    - output from the network
    """
    # Calculate outputs for the hidden layer neurons
    n3_output = gH(x1 * weights["w1"] + x2 * weights["w2"])
    n4_output = gH(x1 * weights["w3"] + x2 * weights["w4"])

    # Calculate output for the output layer neuron
    n5_output = gO(n3_output * weights["w5"] + n4_output * weights["w6"])

    return n5_output

# Define the weights for ANN 1, ANN 2, ANN 3, and ANN 4
# Note: The weights are placeholders. You need to fill them based on Figure 12.
ann_weights = {
    "ANN 1": {"w1": 0, "w2": 0, "w3": 0, "w4": 0, "w5": 0, "w6": 0},
    "ANN 2": {"w1": 0, "w2": 0, "w3": 0, "w4": 0, "w5": 0, "w6": 0},
    "ANN 3": {"w1": 0, "w2": 0, "w3": 0, "w4": 0, "w5": 0, "w6": 0},
    "ANN 4": {"w1": 0, "w2": 0, "w3": 0, "w4": 0, "w5": 0, "w6": 0},
}

# Propagate inputs x1=-1 and x2=-1 through each network
x1, x2 = -1, -1
outputs = {}
for ann, weights in ann_weights.items():
    outputs[ann] = propagate_through_network(weights, x1, x2)

# Find the network with output close to zero
correct_ann = [ann for ann, output in outputs.items() if np.isclose(output, 0, atol=1e-4)]
print(f"The artificial neural network that corresponds to the given description is: {correct_ann[0]}")

# 17 may 2017
import numpy as np


def hyperbolic_tangent(x):
    """
    Hyperbolic tangent activation function.

    Parameters:
    - x: input value

    Returns:
    - value of hyperbolic tangent function at x
    """
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def ann_predict(x, w1_1, w1_2, w2_0, w2_1, w2_2):
    """
    Compute the predicted fuel consumption using the ANN model.

    Parameters:
    - x: observation vector
    - w1_1, w1_2: weight vectors for hidden layer units
    - w2_0, w2_1, w2_2: weights for output layer

    Returns:
    - predicted fuel consumption
    """
    # Bias is always 1
    x = np.array([1] + x)

    # Compute the output from the two hidden units
    h1_output = hyperbolic_tangent(np.dot(x, w1_1))
    h2_output = hyperbolic_tangent(np.dot(x, w1_2))

    # Compute the final predicted fuel consumption
    prediction = w2_1 * h1_output + w2_2 * h2_output + w2_0
    return prediction


# Define the weights and observation vector
w1_1 = np.array([-4, 1, 0.01, 1, -1, -1])
w1_2 = np.array([-10, 1, -0.02, 1, 1, 1])
w2_0 = 7
w2_1 = 8
w2_2 = 9
observation_vector = [6, 120, 3.2, 0, 4]

# Compute the predicted fuel consumption
predicted_consumption = ann_predict(observation_vector, w1_1, w1_2, w2_0, w2_1, w2_2)

print(f"The predicted fuel consumption is: {predicted_consumption:.2f}")

# 16 dec 2017
import numpy as np


def relu(x):
    """
    Rectified Linear Unit (ReLU) activation function.

    Parameters:
    - x: input value

    Returns:
    - Value after ReLU operation
    """
    return np.maximum(x, 0)


def ann_score_predict(x, w1_1, w1_2, w2_0, w2_1, w2_2):
    """
    Compute the predicted average score using the ANN model.

    Parameters:
    - x: observation vector
    - w1_1, w1_2: weight vectors for hidden layer units
    - w2_0, w2_1, w2_2: weights for output layer

    Returns:
    - Predicted average score
    """
    # Bias is always 1
    x = np.array([1] + x)

    # Compute the output from the two hidden units
    h1_output = relu(np.dot(x, w1_1))
    h2_output = relu(np.dot(x, w1_2))

    # Compute the final predicted average score
    score = w2_0 + w2_1 * h1_output + w2_2 * h2_output
    return score


# Define the weights and observation vector
w1_1 = np.array([21.78, -1.65, 0, -13.26, -8.46])
w1_2 = np.array([-9.60, -0.44, 0.01, 14.54, 9.50])
w2_0 = 2.84
w2_1 = 3.25
w2_2 = 3.46
observation_vector = [6.8, 225, 0.44, 0.68]

# Compute the predicted average score
predicted_score = ann_score_predict(observation_vector, w1_1, w1_2, w2_0, w2_1, w2_2)

print(f"The predicted average score is: {predicted_score:.2f}")

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
# 9 may 2018
def calculate_total_models(H, outer_folds, inner_folds, initializations):
    """
    Calculate the total number of models trained using two-level cross-validation.

    Parameters:
    - H: Number of hidden units.
    - outer_folds: Number of cross-validation folds in the outer loop.
    - inner_folds: Number of cross-validation folds in the inner loop.
    - initializations: Number of random initializations for each model.

    Returns:
    - Total number of models trained.
    """
    inner_fold_models = inner_folds * H
    total_models_per_outer_fold = inner_fold_models + 1
    return outer_folds * total_models_per_outer_fold * initializations


def find_max_H(budget):
    """
    Find the largest number of hidden units (H) such that the total number of
    models trained is less than or equal to a given computational budget.

    Parameters:
    - budget: The maximum number of models we can train.

    Returns:
    - The maximum number of hidden units (H) for the given budget.
    """
    # Static values for parameters
    outer_folds = 5
    inner_folds = 10
    initializations = 3

    H = 1
    while True:
        total_models = calculate_total_models(H, outer_folds, inner_folds, initializations)
        if total_models > budget:
            return H - 1
        H += 1


# Static budget value
budget = 1000
max_H = find_max_H(budget)
print(f"The largest value of H for which no more than 1000 models will be trained is: {max_H}")

# 22 dec 2018
import numpy as np


def relu(x):
    """
    Rectified Linear Unit (ReLU) activation function.

    Parameters:
    - x: Input value.

    Returns:
    - Activated value using ReLU.
    """
    return max(0, x)


def neural_network_activation(x7, w1, w2, w2_0):
    """
    Computes the activation of a neural network for a given x7.

    Parameters:
    - x7: Feature value.
    - w1: Weights for the hidden layer.
    - w2: Weights for the output layer.
    - w2_0: Bias for the output layer.

    Returns:
    - Activation value of the neural network.
    """
    # Compute hidden layer activations
    n = [relu(np.dot([1, x7], w)) for w in w1]

    # Compute the final output
    output = w2_0 + np.dot(w2, n)
    return output


# Given weights
w1 = np.array([[-1.8, -1.1], [-0.6, 3.8]])
w2 = np.array([-0.1, 2.1])
w2_0 = -0.8

# Compute neural network activation for x7 = 2
x7_value = 2
activation_value = neural_network_activation(x7_value, w1, w2, w2_0)
print(f"The activation of the neural network for x7 = {x7_value} is: {activation_value}")

# Based on the result, we can infer the correct output from the provided options.

# 10 may 2019
import numpy as np


def sigmoid(x):
    """
    Sigmoid activation function.

    Parameters:
    - x: Input value.

    Returns:
    - Activated value using sigmoid function.
    """
    return 1 / (1 + np.exp(-x))


def linear(x):
    """
    Linear activation function.

    Parameters:
    - x: Input value.

    Returns:
    - Activated value using linear function.
    """
    return x


def neural_network_activation(x, w1, w2, w2_0):
    """
    Computes the activation of a neural network for given x values.

    Parameters:
    - x: Feature values [x1, x2].
    - w1: Weights for the hidden layer.
    - w2: Weights for the output layer.
    - w2_0: Bias for the output layer.

    Returns:
    - Activation value of the neural network.
    """
    # Compute hidden layer activations
    n = [sigmoid(np.dot([1] + x, w)) for w in w1]

    # Compute the final output using a linear activation
    output = w2_0 + np.dot(w2, n)
    return linear(output)


# Given weights
w1 = np.array([[-1.2, -1.3, 0.6], [-1.0, -0.0, 0.9]])
w2 = np.array([-0.3, 0.5])
w2_0 = 2.2

# Compute neural network activation for x = [3, 3]
x_values = [3, 3]
activation_value = neural_network_activation(x_values, w1, w2, w2_0)
print(f"The activation of the neural network for x = {x_values} is: {activation_value:.3f}")
# Based on the result, we can infer the correct output from the provided options.

# 25 dec 2020
def compute_parameters(input_features, hidden_units, classes):
    """
    Compute the total number of parameters in the neural network.

    Parameters:
    - input_features: Number of input features (M).
    - hidden_units: Number of units in the hidden layer (nh).
    - classes: Number of output classes (C).

    Returns:
    - Total number of trainable parameters in the network.
    """

    # Parameters between input and hidden layer: Each hidden unit has input_feature weights and 1 bias.
    input_to_hidden_params = hidden_units * (input_features + 1)

    # Parameters between hidden layer and softmax layer: Each class will have hidden_units weights and 1 bias.
    hidden_to_output_params = classes * (hidden_units + 1)

    # Total parameters
    total_parameters = input_to_hidden_params + hidden_to_output_params

    return total_parameters


# Given values
M = 4  # Number of input features
nh = 6  # Number of hidden units
C = 3  # Number of classes

# Compute total parameters
parameters = compute_parameters(M, nh, C)
print(f"The neural network contains {parameters} parameters.")

# 17 dec 2021
def compute_parameters(input_features, hidden_units, classes):
    """
    Compute the total number of parameters in the neural network.

    Parameters:
    - input_features (int): Number of input features (M).
    - hidden_units (int): Number of units in the hidden layer (nh).
    - classes (int): Number of output classes (C).

    Returns:
    - int: Total number of trainable parameters in the network.
    """

    # Parameters between input and hidden layer:
    # Each hidden unit has input_feature weights and 1 bias.
    input_to_hidden_params = hidden_units * (input_features + 1)

    # Parameters between hidden layer and softmax layer:
    # Each class will have hidden_units weights and 1 bias.
    hidden_to_output_params = classes * (hidden_units + 1)

    # Total parameters
    total_parameters = input_to_hidden_params + hidden_to_output_params

    return total_parameters


# Given values
M = 8  # Number of input features
nh = 50  # Number of hidden units
C = 9  # Number of classes

# Compute total parameters
parameters = compute_parameters(M, nh, C)
print(f"The neural network contains {parameters} parameters.")
