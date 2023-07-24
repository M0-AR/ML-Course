"""
02450ex_Fall2021_sol-1.pdf
Question 17
"""
# Define the number of features, hidden units, and classes
M, nh, C = 8, 50, 9

# Calculate the number of parameters
params_hidden = (M + 1) * nh
params_output = (nh + 1) * C
total_params = params_hidden + params_output

print("Total number of parameters: ", total_params)
"""
02450ex_Fall2021_sol-1.pdf
Question 18
"""
import numpy as np

# Define the ReLU function
def relu(x):
    return np.maximum(0, x)

# Define the weights for the first layer
w1 = np.array([-2, 4, 2])

# Define the input data
X = np.array([
    [1, 0, 0],
    [1, 1, 0],
    [1, 1, 1],
    [1, 1, 2]
])

# Calculate the output of the first layer
Z1 = relu(np.dot(X, w1))

# Now we have a simple linear regression problem: we want to predict y from Z1
y = np.array([1, 3, 5, 7])

# Append a column of ones to Z1 to account for the bias term in the second layer
Z1 = np.column_stack([np.ones(Z1.shape[0]), Z1])

# Calculate the weights for the second layer using the normal equation
w2 = np.linalg.inv(Z1.T.dot(Z1)).dot(Z1.T).dot(y)

print("Weights for the second layer: ", w2)