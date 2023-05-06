"""
Created by Mohamad-s176492
"""

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
from scipy.linalg import svd

# Subtract mean value from data
Y = X - np.ones((N, 1)) * X.mean(axis=0)

# PCA by computing SVD of Y
U, S, V = svd(Y, full_matrices=False)

# Compute variance explained by principal components
rho = (S * S) / (S * S).sum()

threshold = 0.9

# Plot variance explained
plt.figure()
plt.plot(range(1, len(rho) + 1), rho, 'x-')
plt.plot(range(1, len(rho) + 1), np.cumsum(rho), 'o-')
plt.plot([1, len(rho)], [threshold, threshold], 'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual', 'Cumulative', 'Threshold'])
plt.grid()
plt.show()

print('From Exercise 2.1.3')


# Plot the first two principal directions as arrows
origin = np.zeros(X.shape[1])
plt.figure()
for i in range(2):
    plt.arrow(origin[0], origin[1], V[i,0], V[i,1], color='r', head_width=0.1)
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Principal Directions of PCA Components')
plt.show()


from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, show, legend

# scipy.linalg.svd returns "Vh", which is the Hermitian (transpose)
# of the vector V. So, for us to obtain the correct V, we transpose:
V = V.T

# Project the centered data onto principal component space
Z = Y @ V

# Indices of the principal components to be plotted
i = 0
j = 1

# Plot PCA of the data
f = figure()
title('Heart Diseases data: PCA')
# Z = array(Z)
for c in range(C):
    # select indices belonging to class c:
    class_mask = y == c
    plot(Z[class_mask, i], Z[class_mask, j], 'o', alpha=.5)
legend(classNames)
xlabel('PC{0}'.format(i + 1))
ylabel('PC{0}'.format(j + 1))

# Output result to screen
show()

print('From Exercise 2.1.4')

# We saw in 2.1.3 that the first 3 components explaiend more than 90
# percent of the variance. Let's look at their coefficients:
pcs = [0, 1, 2]
legendStrs = ['PC' + str(e + 1) for e in pcs]
c = ['r', 'g', 'b']
bw = .2
r = np.arange(1, M + 1)
for i in pcs:
    plt.bar(r + i * bw, V[:, i], width=bw)
plt.xticks(r + bw, attributeNames)
plt.xlabel('Attributes')
plt.ylabel('Component coefficients')
plt.legend(legendStrs)
plt.grid()
plt.title('Heart Diseases: PCA Component Coefficients')
plt.show()


# Plot a bar chart of the attribute standard deviations
r = np.arange(1, X.shape[1] + 1)
plt.bar(r, np.std(X, 0))
plt.xticks(r, attributeNames)
plt.ylabel('Standard deviation')
plt.xlabel('Attributes')
plt.title('Heart Disease: attribute standard deviations')
plt.show()

# Subtract the mean from the data
Y1 = X - np.ones((N, 1)) * X.mean(0)

# Subtract the mean from the data and divide by the attribute standard
# deviation to obtain a standardized dataset:
Y2 = X - np.ones((N, 1)) * X.mean(0)
Y2 = Y2 * (1 / np.std(Y2, 0))
# Here were utilizing the broadcasting of a row vector to fit the dimensions
# of Y2

# Store the two in a cell, so we can just loop over them:
Ys = [Y1, Y2]
titles = ['Zero-mean', 'Zero-mean and unit variance']
threshold = 0.9
# Choose two PCs to plot (the projection)
i = 0
j = 1

# Make the plot
plt.figure(figsize=(10, 15))
plt.subplots_adjust(hspace=.4)
plt.title('NanoNose: Effect of standardization')
nrows = 3
ncols = 2
for k in range(2):
    # Obtain the PCA solution by calculate the SVD of either Y1 or Y2
    U, S, Vh = svd(Ys[k], full_matrices=False)
    V = Vh.T  # For the direction of V to fit the convention in the course we transpose
    # For visualization purposes, we flip the directionality of the
    # principal directions such that the directions match for Y1 and Y2.
    if k == 1: V = -V; U = -U;

    # Compute variance explained
    rho = (S * S) / (S * S).sum()

    # Compute the projection onto the principal components
    Z = U * S;

    # Plot projection
    plt.subplot(nrows, ncols, 1 + k)
    C = len(classNames)
    for c in range(C):
        plt.plot(Z[y == c, i], Z[y == c, j], '.', alpha=.5)
    plt.xlabel('PC' + str(i + 1))
    plt.ylabel('PC' + str(j + 1))
    plt.title(titles[k] + '\n' + 'Projection')
    plt.legend(classNames)
    plt.axis('equal')

    # Plot attribute coefficients in principal component space
    plt.subplot(nrows, ncols, 3 + k)
    for att in range(V.shape[1]):
        plt.arrow(0, 0, V[att, i], V[att, j])
        plt.text(V[att, i], V[att, j], attributeNames[att])
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.xlabel('PC' + str(i + 1))
    plt.ylabel('PC' + str(j + 1))
    plt.grid()
    # Add a unit circle
    plt.plot(np.cos(np.arange(0, 2 * np.pi, 0.01)),
             np.sin(np.arange(0, 2 * np.pi, 0.01)));
    plt.title(titles[k] + '\n' + 'Attribute coefficients')
    plt.axis('equal')

    # Plot cumulative variance explained
    plt.subplot(nrows, ncols, 5 + k);
    plt.plot(range(1, len(rho) + 1), rho, 'x-')
    plt.plot(range(1, len(rho) + 1), np.cumsum(rho), 'o-')
    plt.plot([1, len(rho)], [threshold, threshold], 'k--')
    plt.title('Variance explained by principal components');
    plt.xlabel('Principal component');
    plt.ylabel('Variance explained');
    plt.legend(['Individual', 'Cumulative', 'Threshold'])
    plt.grid()
    plt.title(titles[k] + '\n' + 'Variance explained')

plt.show()

# We saw in 2.1.3 that the first 7 components explaiend more than 90
# percent of the variance. Let's look at their coefficients:
pcs = [0, 1, 2, 3, 4, 5, 6]
legendStrs = ['PC' + str(e + 1) for e in pcs]
c = ['r', 'g', 'b']
bw = .2
r = np.arange(1, M + 1)
for i in pcs:
    plt.bar(r + i * bw, V[:, i], width=bw)
plt.xticks(r + bw, attributeNames)
plt.xlabel('Attributes')
plt.ylabel('Component coefficients')
plt.legend(legendStrs)
plt.grid()
plt.title('Heart Diseases: PCA Component Coefficients')
plt.show()


# # Use the first two principal components
# pcas = [0, 1]
#
# # Project the data onto the first two principal components
# Z = X_pca[:, pcas]
#
# # Create scatter plot
# plt.scatter(Z[y==0, 0], Z[y==0, 1], c='blue', label='No Heart Disease')
# plt.scatter(Z[y==1, 0], Z[y==1, 1], c='orange', label='Heart Disease')
# plt.xlabel('PC{}'.format(pcas[0] + 1))
# plt.ylabel('PC{}'.format(pcas[1] + 1))
# plt.legend()
# plt.show()


# Plot the first two principal directions as arrows
origin = np.zeros(X.shape[1])
plt.figure()
for i in range(2):
    plt.arrow(origin[0], origin[1], V[i,0], V[i,1], color='r', head_width=0.1)
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Principal Directions of PCA Components')
plt.show()

print("From Exercise 2.1.6")

# -------------------------------- VIS -------------------------------------------
# exercise 4.3.1

from matplotlib.pyplot import (figure, title, boxplot, xticks, subplot, hist,
                               xlabel, ylim, yticks, show)
import numpy as np
from scipy.io import loadmat
from scipy.stats import zscore

attributeNames = data.columns[1:10].tolist()

# We start with a box plot of each attribute
figure()
title('Heart disease: Boxplot')
boxplot(X)
xticks(range(1, M+1), attributeNames, rotation=45)

# From this plot, it is difficult to see the distribution of the data because the axis is dominated by extreme outliers.
# To avoid this, we plot a box plot of standardized data (using the zscore function).
figure(figsize=(12, 6))
title('Heart disease: Boxplot (standardized)')
boxplot(zscore(X, ddof=1), attributeNames)
xticks(range(1, M+1), attributeNames, rotation=45)

# This plot reveals that there are no clear outliers in the dataset.

# Next, we plot histograms of all attributes.
figure(figsize=(14, 9))
u = int(np.floor(np.sqrt(M))); v = int(np.ceil(float(M)/u))
for i in range(M):
    subplot(u, v, i+1)
    hist(X[:, i])
    xlabel(attributeNames[i])
    ylim(0, N) # Make the y-axes equal for improved readability
    if i % v != 0:
        yticks([])
    if i == 0:
        title('Heart disease: Histogram')

show()

print('From Exercise 4.3.1')


Xnorm = zscore(X, ddof=1)

## Next we plot a number of atttributes
Attributes = [0,1,2,3,4,5,6,7]
NumAtr = len(Attributes)

figure(figsize=(12,12))
for m1 in range(NumAtr):
    for m2 in range(NumAtr):
        subplot(NumAtr, NumAtr, m1*NumAtr + m2 + 1)
        for c in range(C):
            class_mask = (y==c)
            plot(X[class_mask,Attributes[m2]], X[class_mask,Attributes[m1]], '.')
            if m1==NumAtr-1:
                xlabel(attributeNames[Attributes[m2]])
            else:
                xticks([])
            if m2==0:
                ylabel(attributeNames[Attributes[m1]])
            else:
                yticks([])
            #ylim(0,X.max()*1.1)
            #xlim(0,X.max()*1.1)
legend(classNames)
show()

print('From Exercise 4.3.2')

# Plot PCA of the data
f = figure()
title('Heart Diseases data: PCA')
# Z = array(Z)
for c in range(C):
    # select indices belonging to class c:
    class_mask = y == c
    plot(Z[class_mask, i], Z[class_mask, j], 'o', alpha=.5)
legend(classNames)
xlabel('PC{0}'.format(i + 1))
ylabel('PC{0}'.format(j + 1))

# Output result to screen
show()

# --------------------------------------------------
# Boxplot (standardized) + histograms
# --------------------------------------------------
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

# exercise 4.3.1 + 4.3.2
from matplotlib.pyplot import (figure, title, boxplot, xticks, subplot, hist,
                               xlabel, ylim, yticks, show)
import numpy as np
from scipy.stats import zscore

attributeNames = data.columns[1:10].tolist()

# We start with a box plot of each attribute
figure()
title('Heart disease: Boxplot')
boxplot(X)
xticks(range(1, M+1), attributeNames, rotation=45)


# From this plot, it is difficult to see the distribution of the data because the axis is dominated by extreme outliers.
# To avoid this, we plot a box plot of standardized data (using the zscore function).
figure(figsize=(12, 6))
title('Heart disease: Boxplot (standardized)')
boxplot(zscore(X, ddof=1), attributeNames)
xticks(range(1, M+1), attributeNames, rotation=45)

# This plot reveals that there are no clear outliers in the dataset.

# Next, we plot histograms of all attributes.
figure(figsize=(14, 9))
u = int(np.floor(np.sqrt(M))); v = int(np.ceil(float(M)/u))
for i in range(M):
    subplot(u, v, i+1)
    hist(X[:, i])
    xlabel(attributeNames[i])
    ylim(0, N) # Make the y-axes equal for improved readability
    if i % v != 0:
        yticks([])
    if i == 0:
        title('Heart disease: Histogram')

show()


# Define the domain for each attribute
domain = {
    'sbp': [101, 218],
    'tobacco': [0.0, 31.2],
    'ldl': [0.98, 15.33],
    'adiposity': [6.74, 42.49],
    'famhist': [0, 1],
    'typea': [13, 78],
    'obesity': [14.7, 46.58],
    'alcohol': [0.0, 147.19],
    'age': [15, 64],
    'CLASS': [-1, 1]
}


# Using attribute range method to detect outliers and remove them from the dataset
outlier_mask = (X[:,0]>218) | (X[:,1]>31.2) | (X[:,2]>15.33) | (X[:,5]<13) | (np.logical_or(X[:,6]<14.7, X[:,6]>46.58)) | (X[:, 7]>147.19)
valid_mask = np.logical_not(outlier_mask)

# Finally we will remove these from the data set
X = X[valid_mask,:]
y = y[valid_mask]
N = len(y)

# https://absentdata.com/python/how-to-find-outliers-in-python/
# Using the IQR method to detect outliers and remove them from the dataset
Q1 = np.percentile(X, 25, axis=0)
Q3 = np.percentile(X, 75, axis=0)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outlier_mask = (X < lower_bound) | (X > upper_bound)
valid_mask = np.logical_not(np.any(outlier_mask, axis=1))

# print the name of the attributes that outlier seems to be appeared from
outlier_attribute_names = []
for i, attribute_name in enumerate(attributeNames):
    if np.any(outlier_mask[:, i]):
        outlier_attribute_names.append(attribute_name)
print("Attributes with outliers:", outlier_attribute_names)

# Finally we will remove these from the data set
X = X[valid_mask,:]
y = y[valid_mask]
N = len(y)


# Now, we can repeat the process to see if there are any more outliers
# present in the data. We take a look at a histogram of all attributes:
figure(figsize=(14,9))
u = int(np.floor(np.sqrt(M))); v = int(np.ceil(float(M)/u))
for i in range(M):
    subplot(u,v,i+1)
    hist(X[:,i])
    xlabel(attributeNames[i])
    ylim(0, N) # Make the y-axes equal for improved readability
    if i%v!=0: yticks([])
    if i==0: title('Heart disease: Histogram (after outlier detection)')

# This reveals no further outliers, and we conclude that all outliers have
# been detected and removed.

show()

print('From Exercise 4.3.1 + 4.3.2')

# --------------------------------------------------
# Plot scatter plot for each combination of PC pairs
# --------------------------------------------------

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