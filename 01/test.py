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


# TODO: need to change the description below to satisfy our attributes domain
# The histograms show that there are a few very extreme values in these
# three attributes. To identify these values as outliers, we must use our
# knowledge about the data set and the attributes. Say we expect volatide
# acidity to be around 0-2 g/dm^3, density to be close to 1 g/cm^3, and
# alcohol percentage to be somewhere between 5-20 % vol. Then we can safely
# identify the following outliers, which are a factor of 10 greater than
# the largest we expect.

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

from matplotlib.pyplot import figure, subplot, plot, legend, show, xlabel, ylabel, xticks, yticks, hist, title, bar, \
    boxplot
import numpy as np
from scipy.stats import zscore
import seaborn as sns
import matplotlib.pyplot as plt

# Normalize input variables
Xnorm = zscore(X, ddof=1)

# Calculate correlation matrix
R = np.corrcoef(Xnorm.T)

# Create a figure with a larger height
fig, ax = plt.subplots(figsize=(10, 8))

# Plot correlation matrix as a heatmap
figure()
sns.heatmap(R, annot=True, xticklabels=attributeNames, yticklabels=attributeNames, ax=ax)
show()

# Calculate correlation coefficients between target variable and attributes
corr_coeffs = np.zeros(M)
for i in range(M):
    corr_coeffs[i] = np.corrcoef(y, X[:, i])[0, 1]

# Plot bar chart of correlation coefficients
figure()
bar_positions = np.arange(M)
bar_heights = corr_coeffs
xticks(bar_positions, attributeNames, rotation=45)
ylabel('Correlation Coefficient')
bar(bar_positions, bar_heights)
show()