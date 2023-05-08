######################################
# May 2018
######################################
print('May 2018')
# 1
"""
Based on the information provided in Table 1 and without being able to see the matrix plot in Figure 1, we can deduce the following:

A. We cannot confidently conclude if at least one of the attributes appears to be normally distributed without seeing the matrix plot in Figure 1.
B. Without seeing the matrix plot in Figure 1, we cannot determine if x2 and x3 are negatively correlated.
C. All six attributes are ratio since they have a true zero point and can be meaningfully compared using division.
D. All the attributes are continuous since they can take on any value within a specific range.

Therefore, based on the information available, the correct answer is:

C. All the six attributes are ratio.


When I say that the attributes have a "true zero point" and "can be meaningfully compared using division," I am referring to the characteristics of ratio data.

Ratio data is a type of data that has all the properties of interval data (i.e., it can be added, subtracted, and the difference between two points is meaningful), but it also has a true zero point. The true zero point means that the value of zero represents the complete absence of the attribute being measured. For example, in the case of the airline safety dataset, zero fatalities would mean no deaths occurred during the given time period.

In addition to having a true zero point, ratio data can also be meaningfully compared using division. This means that we can compare two values by dividing them, and the resulting ratio has a meaningful interpretation. For example, if one airline has 100 fatalities and another has 50 fatalities during the same time period, we can say that the first airline has twice as many fatalities as the second one (100/50 = 2).

In conclusion, when I mention that the attributes have a true zero point and can be meaningfully compared using division, I am highlighting the characteristics of ratio data, which applies to all six attributes in the airline safety dataset.
"""

# 2
"""
To determine the correct variance explained curve, we will first calculate the proportion of variance explained by each principal component using the singular values in matrix S. We can calculate the total variance by squaring each singular value and summing them up:

Total Variance = 13.5^2 + 7.6^2 + 6.5^2 + 5.8^2 + 3.5^2 + 2.0^2

Next, we can calculate the cumulative proportion of variance explained by each principal component:

Proportion of variance explained by the 1st component = (13.5^2) / Total Variance
Proportion of variance explained by the first two components = (13.5^2 + 7.6^2) / Total Variance
Proportion of variance explained by the first three components = (13.5^2 + 7.6^2 + 6.5^2) / Total Variance
Proportion of variance explained by the first four components = (13.5^2 + 7.6^2 + 6.5^2 + 5.8^2) / Total Variance
Proportion of variance explained by the first five components = (13.5^2 + 7.6^2 + 6.5^2 + 5.8^2 + 3.5^2) / Total Variance
Proportion of variance explained by all six components = Total Variance / Total Variance = 1
"""
import numpy as np

# Singular values from matrix S
singular_values = np.array([13.5, 7.6, 6.5, 5.8, 3.5, 2.0])

# Calculate the total variance
total_variance = np.sum(singular_values ** 2)

# Calculate the cumulative proportion of variance explained
cumulative_variance_explained = np.cumsum(singular_values ** 2) / total_variance

print('2018-may-2')
# Print the cumulative proportion of variance explained
print("Cumulative Variance Explained: ", cumulative_variance_explained)

# Provided curves
curve_1 = np.array([0.5, 0.7, 0.85, 0.95, 0.98, 1])
curve_2 = np.array([0.35, 0.55, 0.7, 0.85, 0.95, 1])
curve_3 = np.array([0.22, 0.42, 0.6, 0.8, 0.9, 1])
curve_4 = np.array([0.72, 0.85, 0.92, 0.97, 1, 1])

# Find the closest curve
curves = [curve_1, curve_2, curve_3, curve_4]
min_diff = float('inf')
correct_curve = -1

for i, curve in enumerate(curves, 1):
    diff = np.sum(np.abs(curve - cumulative_variance_explained))
    if diff < min_diff:
        min_diff = diff
        correct_curve = i

# Print the correct curve
print(f"Correct Curve: Variance Explained Curve {correct_curve}")
print('\n')

# 3
"""
The question asks you to project a standardized observation onto the first two principal components using the PCA directions given by the matrix V.

The standardized observation is: ̃x∗ = [-0.1, 0.2, 0.1, -0.3, 1, 0.5]

The PCA directions are given in matrix V. The first two columns of V represent the first two principal components. To project the standardized observation onto these components, you need to multiply the observation by the first two columns of V.

In theory:

Extract the first two columns from matrix V.
Multiply the standardized observation by the extracted columns.
The result will give you the coordinates of the observation when projected onto the first two principal components.

Fact:

When working with high-dimensional data, it is often desirable to reduce the number of dimensions to make the data easier to visualize, analyze, and process. One common technique for dimensionality reduction is Principal Component Analysis (PCA).

PCA finds a set of new axes, called principal components, that maximize the variance of the data when projected onto them. These principal components are linear combinations of the original variables and are orthogonal to each other. In other words, they are uncorrelated, and each component captures different information present in the data. The first principal component captures the highest amount of variance in the data, the second principal component captures the next highest amount of variance, and so on.

By projecting the data onto the first few principal components, we can reduce the number of dimensions while preserving most of the information in the data. This can help in data visualization, pattern recognition, and understanding the underlying structure of the data.

In this question, you are given a standardized observation, ̃x∗ = [-0.1, 0.2, 0.1, -0.3, 1, 0.5]. You are asked to project this observation onto the first two principal components. This projection will give you a new representation of the observation in a lower-dimensional space (2D) that still captures most of the variance in the data. By doing this, you can understand the relationships between variables better, visualize the data more effectively, and potentially improve the performance of machine learning algorithms on the dataset.

In summary, projecting an observation onto the first two principal components allows you to:

Reduce the dimensionality of the data.
Preserve the maximum amount of variance in the data.
Visualize the data more easily and recognize patterns.
Potentially improve the performance of machine learning algorithms.
"""
import numpy as np

# Standardized observation
x_star = np.array([-0.1, 0.2, 0.1, -0.3, 1, 0.5])

# Matrix V
V = np.array([
    [0.38, -0.51, 0.23, 0.47, -0.55, 0.11],
    [0.41, 0.41, -0.53, 0.24, 0.00, 0.58],
    [0.50, 0.34, -0.13, 0.15, -0.05, -0.77],
    [0.29, 0.48, 0.78, -0.17, 0.00, 0.23],
    [0.45, -0.42, 0.09, 0.03, 0.78, 0.04],
    [0.39, -0.23, -0.20, -0.82, -0.30, 0.04]
])

# Extract the first two columns of V
V_first_two = V[:, :2]

# Multiply the standardized observation by the first two columns of V
projection = x_star.dot(V_first_two)

print('2018-may-3')
# Print the result
print(f"Projected coordinates: ({projection[0]:.3f}, {projection[1]:.3f})")
print('\n')

# 4
"""
Question 4. Which one of the following statements
regarding Gaussian Mixture Modeling (GMM) is cor-
rect?

B. For high-dimensional data, i.e., where the number of features M is large, it can be beneficial to constrain the covariance of each cluster to be diagonal, i.e., enforcing off-diagonal terms of the covariance matrices to be zero, in order to reduce the number of parameters in the GMM model.

This statement is correct because, in high-dimensional data, unconstrained covariance matrices can lead to overfitting and require a large number of parameters to be estimated. By constraining the covariance matrices to be diagonal, you reduce the number of parameters to estimate and decrease the chances of overfitting. This can lead to a more robust and efficient Gaussian Mixture Model.

A. The number of clusters used in the GMM can be determined by selecting the number of clusters that provides the best likelihood of the training data used for training the density.

This statement is incorrect because selecting the number of clusters based solely on the likelihood of the training data can lead to overfitting. A better approach would be to use model selection criteria such as the Akaike Information Criterion (AIC) or the Bayesian Information Criterion (BIC), which consider both the likelihood and the complexity of the model.

C. The GMM is guaranteed to find the optimal clustering for a given dataset.

This statement is incorrect because the Expectation-Maximization (EM) algorithm used in GMM is sensitive to the initial parameters and can get stuck in local optima. Thus, GMM does not guarantee finding the global optimal clustering for a given dataset.

D. Similar to the k-means algorithm that assigns observations to the cluster in closest proximity, the EM-algorithm used to estimate the parameters of the GMM considers only the cluster each observation is the most likely to belong to when estimating the parameters in the M-step.

This statement is incorrect because, unlike the k-means algorithm, the EM-algorithm used in GMM considers the probabilities of each observation belonging to each cluster during the M-step, not just the most likely cluster. The GMM is a soft clustering method that assigns each observation a probability of belonging to each cluster, while k-means is a hard clustering method that assigns each observation to only one cluster.
"""

# 5
"""
Look at Figure 3
In the Gaussian Mixture Model (GMM) density equations provided, each term represents a Gaussian distribution (also known as a normal distribution) weighted by a mixing coefficient. The GMM is a weighted sum of these Gaussian distributions. Let's break down the elements of one term in the GMM:

0.0673 · N(x|[1.8422, 2.4306], [[0.2639, 0.0803], [0.0803, 0.0615]])

0.0673: This is the mixing coefficient (weight) of the Gaussian distribution in the GMM. It represents the proportion of the overall GMM accounted for by this Gaussian distribution.

N(x|[1.8422, 2.4306], [[0.2639, 0.0803], [0.0803, 0.0615]]): This is a multivariate Gaussian distribution, which is specified by its mean vector and covariance matrix.

[1.8422, 2.4306]: This is the mean vector (μ) of the Gaussian distribution. It represents the center of the Gaussian distribution in the 2-dimensional space. Each value in the vector corresponds to the mean of a variable (in this case, the first two principal components).

[[0.2639, 0.0803], [0.0803, 0.0615]]: This is the covariance matrix (Σ) of the Gaussian distribution. It characterizes the shape and orientation of the Gaussian distribution. The diagonal elements represent the variances of the variables, while the off-diagonal elements represent the covariances between the variables.

To describe the figure based on the GMM density equations, look at the mixing coefficients, mean vectors, and covariance matrices for each Gaussian distribution in the GMM:

Mixing coefficients: These tell you the relative size of each cluster. A larger mixing coefficient indicates a larger proportion of the data belonging to that cluster.
Mean vectors: These indicate the centers of the clusters. Compare the positions of the centers in the figure with the mean vectors in the GMM density equations.
Covariance matrices: These describe the shape and orientation of the clusters. Look at the variances (diagonal elements) and covariances (off-diagonal elements) to determine which clusters are elongated, rotated, or circular.
By comparing these elements with the figure, you can identify which GMM density matches the figure.

The solution is trying to identify which GMM density equation matches the figure provided. It does this by examining the properties of the Gaussian distributions (clusters) in the GMM.

There are two main observations made in the solution:

The cluster centered at [1.8422, 2.4306] has a low mixing proportion (few observations belong to this cluster) and has a large positive correlation and variance. Only answer options 2 and 4 have this property.

The cluster centered at [1.6359, -0.0183] has a negative covariance and its variance is larger than the variance of the other cluster with negative covariance. This cluster must have the covariance [[4.0475, -1.5818], [-1.5818, 1.1146]]. Only answer option 4 has this property.

By combining these observations, the solution concludes that answer option 4 is the correct one, as it is the only option that satisfies both conditions.

In summary, the solution is comparing the properties of the clusters in the figure with those in the GMM density equations to find the one that matches. In this case, answer option 4 is identified as the correct match.
"""

# 6
"""
Based on the provided solution, we can see the following results from forward and backward selection:

Forward Selection:

    Select x6 with a performance of 0.18749
    Select x3 with a performance of 0.17624
    Select x5 with a performance of 0.17082

No further improvement with additional features, so the final feature set is {x3, x5, x6}
Backward Selection:

    Remove x4 to have the feature set {x2, x3, x5, x6} with a performance of 0.17299
    Remove x2 to have the feature set {x3, x5, x6} with a performance of 0.17082
    No further improvement by removing more features, so the final feature set is {x3, x5, x6}
    The final feature set is the same for both forward and backward selection, which is {x3, x5, x6}. Therefore, the correct statement is:

A. Forward and backward selection will result in the same features being selected.
"""

# 7
"""
The correct statement regarding the described regularized least squares regression procedure is:

C. The test error obtained for the optimal value of λ is a biased estimate of the generalization error.

Explanation:
A. Increasing λ will result in a decrease in the 2-norm of the trained w, not an increase. This is because a larger λ will impose a stronger penalty on larger weights, causing the algorithm to prefer smaller weights.
B. 10-fold cross-validation with 20 different values of λ will require the fitting of 200 models in total (10 folds * 20 values of λ), not just 10 models.
D. Regularization in least squares regression actually helps to prevent overfitting by adding a penalty term that encourages smaller weights. This reduces the complexity of the model, making it less prone to overfitting.
"""

# 8
"""
The solution for question 8 is trying to find which network model from Figure 4 corresponds to the given 
artificial neural network (ANN) with the specified weights. To do this, the solution calculates the output of the ANN 
for two specific points (x5 = 0, x6 = 3) and (x5 = 24, x6 = 0) and then looks for a network model in Figure 4 that 
has similar output values for these points. 

Running this code will give you the output values for the specified points (x5 = 0, x6 = 3) and (x5 = 24, x6 = 0), 
which are 3.4667e-7 and 3.5900e-8, respectively. These output values match the properties of Network Model 4 in 
Figure 4, so the correct answer is Network Model 4 (option D). 
"""
import numpy as np

def logistic(z):
    return 1 / (1 + np.exp(-z))

w1 = np.array([
    [0.0189, 0.9159, -0.4256],
    [3.7336, -0.8003, 5.0741]
])

w2 = np.array([0.3799e-6, -0.3440e-6, 0.0429e-6])

x1 = np.array([1, 0, 3])
x2 = np.array([1, 24, 0])

hidden_output1 = logistic(x1.dot(w1.T))
hidden_output2 = logistic(x2.dot(w1.T))

output1 = w2.dot(np.hstack(([1], hidden_output1)))
output2 = w2.dot(np.hstack(([1], hidden_output2)))

print('2018-may-8')
print(f"Output for x5 = 0, x6 = 3: {output1}")
print(f"Output for x5 = 24, x6 = 0: {output2}")
print('\n')

# 9
"""
To determine the largest value of H for which no more than 1000 models will be trained, let's analyze the training process.

For each value of H (the number of hidden units), we will train 3 models with different random initializations. We will use 10-fold cross-validation in the inner fold to select the optimal number of hidden units and 5-fold cross-validation in the outer fold to estimate the generalization performance.

So for each value of H, we train:

3 models (different initializations) * 10 (inner fold) * 5 (outer fold) = 150 models

We have a budget of 1000 models, so we want to find the largest integer value of H such that:

150 * H <= 1000

Dividing both sides by 150:

H <= 1000 / 150

H <= 6.666...

Since H must be an integer, the largest value of H is 6.

So the answer is A. 6.
"""