# 19 2023

import numpy as np

# Given data
x = np.array([-0.5, 0.39, 1.19, -1.08])
yr = np.array([-0.86, -0.61, 1.37, 0.1])

# Learned weights
w_learned = np.array([0.39, 0.77])

# Standardization function
def standardize(data):
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0, ddof=1)

# Transformation options
def option_A(x):
    return np.column_stack([x, x**3])

def option_B(x):
    return np.column_stack([x, np.sin(x), x**2])

def option_C(x):
    return np.column_stack([x, np.sin(x)])

def option_D(x):
    return np.column_stack([x, x**2])

# Apply transformations and standardization
options = {
    'A': standardize(option_A(x)),
    'B': standardize(option_B(x)),
    'C': standardize(option_C(x)),
    'D': standardize(option_D(x))
}

# Iterate over the set of transformation options to identify which
# transformation produces weights similar to our known learned weights (w_learned).
for key, value in options.items():

    # Check if the number of features (columns) in the transformed data matrix (value)
    # matches the length of the known learned weights (w_learned).
    # This ensures we can validly perform a least squares computation for this transformation.
    if value.shape[1] == len(w_learned):

        # Estimate the weights for this transformation using the least squares method.
        # We try to find weights that would fit the transformed data (value) to the
        # known output vector (yr).
        # 'np.linalg.lstsq' returns multiple values, but we're only interested
        # in the actual weight values, hence the [0] indexing.
        w_estimated = np.linalg.lstsq(value, yr, rcond=None)[0]

        # Check if the estimated weights (w_estimated) are approximately equal to the
        # learned weights (w_learned) within an absolute tolerance of 0.1.
        # This helps us identify if the current transformation in consideration
        # is likely the one used to obtain w_learned.
        if np.allclose(w_estimated, w_learned, atol=0.1):
            # If the condition is met, print out the key (transformation identifier)
            # to indicate this transformation is a likely match.
            print(f"Option {key} seems to be the correct transformation.")
