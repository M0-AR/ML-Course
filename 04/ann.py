"""This script is a forward pass of a multi-layer perceptron (MLP) for binary classification. It seems to take a
2-dimensional input vector, applies a transfer function (either sigmoid, ReLU, or tanh), passes the result through a
hidden layer, and then applies a final sigmoid transfer function to yield a single output, the binary class prediction.

Here's a step-by-step description of what this script does:

First, it defines the transfer functions: sigmoid, ReLU, and tanh. These functions are widely used in neural networks
as activation functions. They introduce non-linearity into the model, allowing it to learn complex patterns.

Next, it specifies the initial weights for the model (w0, w1, w2, w_out). The weights w1 and w2 are for the hidden
layer neurons and w_out is for the output neuron. w0 is a bias term.

The weights and the input vector x are packed into numpy arrays.

Depending on the transfer function specified in the transferfunction variable, the respective function (sigmoid,
ReLU, or tanh) is assigned to the variable hFunc.

In the loop, the script goes through each of the hidden layer weights, applies the dot product with the input vector
x, and then applies the chosen transfer function. The results are stored in the list h.

The resulting values in h (outputs of the hidden layer) are then multiplied by the output weights and summed together
with the bias term w0. The sigmoid function is applied to this sum to compute the final output f.

Finally, it prints the output of the network, f.

It's worth noting that this is a simplified script for understanding the basic workings of an MLP. It doesn't include
any training mechanism, which would involve adjusting the weights based on the difference between the output and the
actual target value. This is usually done using a process called backpropagation in combination with a form of
gradient descent. The script also lacks any error checking and robustness features you might expect in a
production-level implementation.
"""
"""
In this revised script:

We've defined a class MLP for the multi-layer perceptron model, with methods for the activation functions (sigmoid, 
ReLU, and tanh) and a prediction function. The __init__ method initializes the MLP with its parameters: the hidden 
layer weights, the output weights, the bias, and the activation function. The predict method carries out the forward 
pass of the MLP: it applies the activation function to the weighted sums of the inputs and the hidden layer weights, 
and then applies the sigmoid function to the weighted sum of the hidden layer outputs. We create an instance of the 
MLP class with the desired parameters and use it to predict the output for the given input vector. This script is 
more modular and maintainable, as the MLP class can be reused for different sets of weights and activation functions. 
The methods within the class can also be tested individually to ensure they're working correctly. """

import numpy as np
""" 
Try the script on:
2016 fall Q20
2018 fall Q22
2019 fall Q20
2021 fall Q18 (difficult setup)
 """

class MLP:
    def __init__(self, hidden_weights, output_weights, bias, activation="sigmoid"):
        self.hidden_weights = np.array(hidden_weights)
        self.output_weights = np.array(output_weights)
        self.bias = bias
        self.activation_function = self.sigmoid if activation == "sigmoid" else self.relu if activation == "relu" else self.tanh

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def relu(x):
        return (x > 0) * x

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    def predict(self, input_vector):
        hidden_layer_output = []
        for weights in self.hidden_weights:
            hidden_layer_output.append(self.activation_function(input_vector @ weights))
        output = self.sigmoid(np.sum(self.output_weights * hidden_layer_output) + self.bias)
        return output


if __name__ == "__main__":
    w0 = 1.4
    w1 = [-0.5, -0.1]
    w2 = [0.9, 2]
    w_out = np.array([-1, 0.4]).transpose()

    x = np.array([1, -2])

    mlp = MLP(hidden_weights=[w1, w2], output_weights=w_out, bias=w0, activation="sigmoid")

    output = mlp.predict(x)

    print(output)
