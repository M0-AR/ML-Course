"""
This code is calculating the gain in data purity that results from a certain split in a decision tree. The context here is multi-class classification, where each of the arrays y0, y1, and y2 represent class distributions.

Let's breakdown the main points:

Impurity Index Calculation (index_regression function): The index_regression function calculates the impurity of a
node in the decision tree for a given set of class labels y. The type of impurity measure calculated depends on the
indexType argument. The different measures include:

'Regression': The mean squared error from the mean. 'Gini': The Gini impurity, which quantifies the probability of
incorrect classification if a new instance were randomly assigned a label from the label distribution in the node.
'Entropy': A measure of impurity used in the creation of decision trees in the ID3/C4.5 algorithms,
based on information theory.
'ClassError': The error rate of the most frequent class. In a binary classification,
for instance, the error is 1 minus the max value of (proportion of positive cases, proportion of negative cases).
Data: The y0, y1, y2 arrays represent the class distribution at the root node (before the split) and at the two child
nodes after the split, respectively.

Impurity Calculation: The function index_regression is called with each of the class distribution arrays to compute
their respective impurities I0, I1, I2 and sizes N0, N1, N2.

Purity Gain Calculation: The purity gain resulting from the split is computed. This is essentially the decrease in
impurity achieved by the split, given by the impurity of the parent node minus the weighted impurities of the child
nodes. The weights are proportional to the sizes of the child nodes relative to the parent node (N1 / N0, N2 / N0).

Output: The calculated purity gain is printed. If this value is positive, it means that the split has resulted in a
reduction in impurity (i.e., an increase in purity). In the context of decision trees, this would be a desirable
split that makes the data more homogeneous with respect to the target variable.

In conclusion, this code is useful in the context of training decision trees, as it helps evaluate potential splits.
A decision tree algorithm would typically perform this calculation for many potential splits, and choose the one with
the greatest purity gain. """
import numpy as np

def calculate_impurity(indexType, y):
    """
    Function to compute the impurity of a node.

    Args:
    indexType (str): The type of impurity index to use ('Regression', 'Gini', 'Entropy', 'ClassError').
    y (list): The list of class counts in the node.

    Returns:
    N (int): The total number of observations in the node.
    I (float): The impurity of the node.
    """
    N = np.sum(y)
    mean = np.mean(y)
    p = y / N

    if indexType == 'Regression':
        N = len(y)
        I = 1 / N * np.sum(np.power((y - mean), 2))

    elif indexType == 'Gini':
        I = 1 - np.sum(np.square(p))

    elif indexType == 'Entropy':
        I = - np.sum(p * np.log2(p))

    elif indexType == 'ClassError':
        I = 1 - np.max(p)

    else:
        raise ValueError("Invalid index type. Use 'Regression', 'Gini', 'Entropy', or 'ClassError'.")

    return N, I

# Initial root (counts of different classes)
y0 = [166, 187, 172]

# Split side one
y1 = [108, 112, 56]

# Split side two
y2 = [58, 75, 116]

# Impurity index type
indexType = 'ClassError'

# Calculate impurity for root and both splits
N0, I0 = calculate_impurity(indexType, y0)
N1, I1 = calculate_impurity(indexType, y1)
N2, I2 = calculate_impurity(indexType, y2)

# Calculate the purity gain from the splits
DeltaPurityGain = I0 - N1 / N0 * I1 - N2 / N0 * I2
print("Purity gain:", DeltaPurityGain)

"""
The purity gain value 0.0780952380952381 is a measure of how much the impurity of the data is reduced by the particular split that has been made.

In the context of building decision trees, a positive purity gain means that the split is separating the data into 
subsets that are more homogeneous with respect to the target variable than the original data. In other words, 
the instances within each subset are more likely to belong to the same class after the split than before. 

A purity gain close to 0 would suggest that the split didn't really do much to separate the classes from one another, 
while a higher value indicates a more effective split. The value 0.0780952380952381 shows a modest improvement in 
purity as a result of the split. 

In practice, when constructing a decision tree, you would compare the purity gains from various possible splits and 
choose the one with the highest gain. This ensures that the tree's splits are chosen in a way that most effectively 
separates the data into clear classes, leading to more accurate predictions on unseen data. """