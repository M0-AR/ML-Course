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
