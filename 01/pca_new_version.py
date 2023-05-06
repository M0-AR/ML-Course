import numpy as np
import pandas as pd

# Load csv file with data
filename = 'saheart_1_withheader.csv'
data = pd.read_csv(filename)

# Extract attribute names (1st row, column 1 to 9)
attributeNames = data.columns[1:10]

# Extract class names to python list,
# then encode with integers (dict)
classLabels = data.iloc[:, 0].values
classNames = sorted(set(classLabels))
classDict = dict(zip(classNames, range(2)))

# Extract vector y, convert to NumPy array
y = np.asarray([classDict[value] for value in classLabels])

# Extract matrix X
X = data.iloc[:, 1:10].values

# Compute values of N, M and C.
N = len(y)
M = X.shape[1]
C = len(classNames)

print('From Exercise 2.1.1')

import matplotlib.pyplot as plt

# Standardize matrix X
Xs = (X - X.mean(axis=0)) / X.std(axis=0)

# Select the first seven principal components
pcas = [0, 1, 2, 3, 4, 5, 6]

# Obtain the PCA solution by calculating the SVD of the standardized data
U, S, Vt = np.linalg.svd(Xs, full_matrices=False)
V = Vt.T

# Compute the principal component scores
Z = Xs.dot(V)

# Plot scatter plot for each combination of PC pairs
for i in range(len(pcas)):
    for j in range(i + 1, len(pcas)):
        # Plot scatter plot for the current PC pair
        plt.scatter(Z[y == 0, pcas[i]], Z[y == 0, pcas[j]], label='No Heart Disease', alpha=0.5)
        plt.scatter(Z[y == 1, pcas[i]], Z[y == 1, pcas[j]], label='Heart Disease', alpha=0.5)

        # Set axis labels and title
        plt.xlabel('PC' + str(pcas[i] + 1))
        plt.ylabel('PC' + str(pcas[j] + 1))
        plt.title('Zero-mean and unit variance Projection')

        # Add legend
        plt.legend()

        # Show the plot
        plt.show()


