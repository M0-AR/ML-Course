# 10
"""
From Table 5, we have:

p(ˆx2 = 0, ˆx4 = 0, y = Machine) = 0.18
p(ˆx2 = 0, ˆx4 = 1, y = Machine) = 0.17
p(ˆx2 = 1, ˆx4 = 0, y = Machine) = 0.08
p(ˆx2 = 1, ˆx4 = 1, y = Machine) = 0.1

We need to calculate:
p(y = Machine | ˆx2 = 1)

Using Bayes' theorem:

p(y = Machine | ˆx2 = 1)
= (p(ˆx2 = 1 | y = Machine) * p(y = Machine)) / p(ˆx2 = 1)

From the table:
p(ˆx2 = 1, y = Machine) = 0.08 + 0.1 = 0.18
p(ˆx2 = 1) = 0.18 + 0.16 = 0.34
p(y = Machine) = 0.18 + 0.17 + 0.08 + 0.1 = 0.53

Therefore:
p(y = Machine | ˆx2 = 1)
= (0.18/0.53) / (0.34)
= 0.34/0.34
= 0.36
"""
probs = {
    (0, 0, 'Machine'): 0.18,
    (0, 1, 'Machine'): 0.17,
    (1, 0, 'Machine'): 0.08,
    (1, 1, 'Machine'): 0.1
}

# p(x2 = 1, y = Machine)
p_x2_1_y = probs[(1, 0, 'Machine')] + probs[(1, 1, 'Machine')]
# 0.08 + 0.1 = 0.18

# p(x2 = 1)
p_x2_1 = 0.18 + 0.16
# 0.34

# p(y = Machine)
p_y = 0.18 + 0.17 + 0.08 + 0.10
# 0.53

# Using Bayes' theorem
p_y_given_x2_1 = (p_x2_1_y / p_y) / p_x2_1

print(p_y_given_x2_1)
print()
# 11
"""
To compute p(y = Machine|x) using a Naive Bayes classifier, we need to evaluate the probability of each feature given the class, i.e., p(xᵢ|y = Machine), assuming that the features are conditionally independent given the class.

Each p(xᵢ|y = Machine) is computed by evaluating the normal density function N(x|μ, σ²) where μ and σ² are the mean and variance of feature xᵢ for y = Machine.

In this case, there are M = 6 features, so we need to evaluate the normal density function 6 times for y = Machine.

However, to compute p(y = Machine|x), we also need to compute p(y = Natural|x) for normalization (because p(y = Machine|x) + p(y = Natural|x) = 1). So, we would need to evaluate the normal density function 6 more times for y = Natural.

Therefore, the minimum number of evaluations of the normal density function N(x|μ, σ²) we have to perform to compute p(y = Machine|x) is 6 (for Machine) + 6 (for Natural) = 12.

So, the answer is C. 12.
"""

# 12
"""
You're right, my previous analysis was incorrect. Let me re-work this from the beginning:

For MA vs MB:
- MB correct, MA correct: 416
- MB wrong, MA wrong: 68
- MB correct, MA wrong: 42
- MB wrong, MA correct: 38

For MA vs MC:
- MC correct, MA correct: 38
- MC wrong, MA wrong: 416
- MC correct, MA wrong: 68
- MC wrong, MA correct: 42

To determine the correct statement, let's recall the formulas for computing the p-values and the difference in accuracy (θ) for McNemar's test.

For two models MA and MB:

p-value: pAB = 2 * cdfT(-|ˆθAB| | ν = 1, μ = 0, σ = 1)
Difference in accuracy: ˆθAB = (n01 - n10) / (n01 + n10)
For two models MA and MC:

p-value: pAC = 2 * cdfT(-|ˆθAC| | ν = 1, μ = 0, σ = 1)
Difference in accuracy: ˆθAC = (n01 - n10) / (n01 + n10)
Looking at the table:

MA correct | MA wrong
MB correct | 416 | 42
MB wrong | 38 | 68

MA correct | MA wrong
MC correct | 68 | 38
MC wrong | 42 | 416

We can compute the values for McNemar's test:

For MA vs. MB:
n01 = 42 (MB correct and MA wrong)
n10 = 38 (MB wrong and MA correct)
ˆθAB = (42 - 38) / (42 + 38) = 4 / 80 = 0.05
pAB = 2 * cdfT(-|0.05| | ν = 1, μ = 0, σ = 1)

For MA vs. MC:
n01 = 38 (MC correct and MA wrong)
n10 = 42 (MC wrong and MA correct)
ˆθAC = (38 - 42) / (38 + 42) = -4 / 80 = -0.05
pAC = 2 * cdfT(-|-0.05| | ν = 1, μ = 0, σ = 1)

Based on the computed values, we find:

pAB = pAC (both p-values are the same)
ˆθAB = -ˆθAC (the difference in accuracy for MA vs. MB is the negative of the difference in accuracy for MA vs. MC)
So, the correct statement is:

A. pAB = pAC and ˆθAB = -ˆθAC.
"""
import scipy.stats as stats

# Define the values from the table
n01_AB = 42  # MB correct and MA wrong
n10_AB = 38  # MB wrong and MA correct

n01_AC = 38  # MC correct and MA wrong
n10_AC = 42  # MC wrong and MA correct

# Compute the differences in accuracy (theta) for MA vs. MB and MA vs. MC
theta_AB = (n01_AB - n10_AB) / (n01_AB + n10_AB)
theta_AC = (n01_AC - n10_AC) / (n01_AC + n10_AC)

# Compute the p-values for MA vs. MB and MA vs. MC
p_AB = 2 * stats.t.cdf(-abs(theta_AB), df=1)
p_AC = 2 * stats.t.cdf(-abs(theta_AC), df=1)

# Print the results
print("p-value for MA vs. MB:", p_AB)
print("Difference in accuracy (theta) for MA vs. MB:", theta_AB)

print("p-value for MA vs. MC:", p_AC)
print("Difference in accuracy (theta) for MA vs. MC:", theta_AC)
print()

# 15
"""Given the information in the question, we could interpret the nodes as having conditions based on the distance
between a given point (x1, x2) and a reference point, using either the L1 or L2 norm. If the distance is less than a
certain threshold, we follow the "True" branch, otherwise we follow the "False" branch.

This interpretation is consistent with the descriptions for each option, where each description appears to indicate a reference point (x1, x2), a norm (1 or 2), and a threshold for each node.
"""
class Node:
    def __init__(self, x1, x2, norm, threshold, true_branch, false_branch):
        self.x1 = x1
        self.x2 = x2
        self.norm = norm
        self.threshold = threshold
        self.true_branch = true_branch
        self.false_branch = false_branch

    def classify(self, observation):
        if self.norm == 1:
            dist = abs(self.x1 - observation[0]) + abs(self.x2 - observation[1])
        else:  # assuming Euclidean norm (norm 2)
            dist = ((self.x1 - observation[0])**2 + (self.x2 - observation[1])**2)**0.5

        if dist < self.threshold:
            return self.true_branch
        else:
            return self.false_branch


# Set up nodes for each option
# Option A
D_A = Node(2, 6, 1, 3, 'class1', 'class2')
B_A = Node(6, 2, 2, 3, D_A, 'class2')
C_A = Node(2, 4, 2, 2, 'class1', 'class2')
A_A = Node(2, 4, 1, 3, B_A, C_A)

# Option B
D_B = Node(2, 4, 2, 2, 'class1', 'class2')
B_B = Node(2, 6, 1, 3, D_B, 'class2')
C_B = Node(2, 4, 1, 3, 'class1', 'class2')
A_B = Node(6, 2, 2, 3, B_B, C_B)

# Option C
D_C = Node(2, 4, 2, 2, 'class1', 'class2')
B_C = Node(2, 6, 1, 3, D_C, 'class2')
C_C = Node(6, 2, 2, 3, 'class1', 'class2')
A_C = Node(2, 4, 1, 3, B_C, C_C)

# Option D
D_D = Node(2, 4, 1, 3, 'class1', 'class2')
B_D = Node(2, 6, 1, 3, D_D, 'class2')
C_D = Node(6, 2, 2, 3, 'class1', 'class2')
A_D = Node(2, 4, 2, 2, B_D, C_D)

# to classify an observation, we would start at Node A:
observation = (3, 3)  # replace with actual values
current_node = A_A
current_node = A_D
while isinstance(current_node, Node):
    current_node = current_node.classify(observation)
print(f"Classified as: {current_node}")

print()
# 17
# Book Cross Validation
# 10.1
# define variables
"""
The total time taken for the entire procedure consists of the time spent on two main tasks: training the model during the cross-validation phase and testing it during the hold-out phase. The regularization strength λ is varied six times during the cross-validation.

First, let's calculate the size of datasets. The validation set Dvalidation is 20% of the total data, which is 0.2 * 1000 = 200. The remainder, used for cross-validation DCV, is then 1000 - 200 = 800.

The time required for cross-validation will be computed as the time needed to train and test the model for each of the ten folds and for each of the six values of the regularization strength. In each fold of the cross-validation, 90% of DCV is used for training (720 observations) and 10% for testing (80 observations).

The time for training in each fold is (720^2) = 518400 units of time. As there are ten folds, the time for training for all folds at a given λ is 51840010 = 5184000 units of time. As λ varies six times, the total training time during the cross-validation phase is 51840006 = 31046400 units of time.

The time for testing in each fold is 1/2*(80^2) = 3200 units of time. As there are ten folds, the time for testing for all folds at a given λ is 320010 = 32000 units of time. As λ varies six times, the total testing time during the cross-validation phase is 320006 = 192000 units of time.

Thus, the total time spent during the cross-validation phase is 31046400 + 192000 = 31238400 units of time.

The hold-out phase involves training the model on DCV (800 observations) and testing it on Dvalidation (200 observations), for the optimal λ found during cross-validation. Thus, the time spent during the hold-out phase is:

Training time: (800^2) = 640000 units of time.
Testing time: 1/2*(200^2) = 20000 units of time.

So the total time spent during the hold-out phase is 640000 + 20000 = 660000 units of time.

In summary, the total time taken for the entire procedure is the sum of the time spent during the cross-validation phase and the hold-out phase, which is 31238400 + 660000 = 31898400 units of time, or approximately 31.90 x 10^6 units of time.

So, the closest answer choice is:

D 31.96 · 10^6 units of time.
"""
"""
This problem might be tricky at first if you're not familiar with cross-validation, hold-out method and the concept of computational cost in terms of training and testing data. Let's break down the problem into simpler steps and guide you through the problem-solving process.

Understanding the problem: The problem is asking you to calculate the total time required for training a linear regression model with different regularization strengths (λ values) and estimating the generalization error using cross-validation and the hold-out method. The time complexity for training is given as N^2 and for testing as 0.5 * N^2.

Splitting the data: The question states that the total data (N=1000) is divided into a validation set (20% of N, which is 200) and the remaining (80% of N, which is 800) is used for cross-validation.

Cross-validation time calculation: Here, the data is split into K folds (K=10), which means for each fold, 90% of data (720 out of 800) is used for training and 10% (80 out of 800) is used for testing. Also, there are 6 possible λ values, so we will repeat this process 6 times.

Training time for each λ value is (720^2)*10 (because there are 10 folds). We multiply the result by 6 for each λ value.

Testing time for each λ value is 0.5 * (80^2)*10 (because there are 10 folds). We multiply the result by 6 for each λ value.

Adding these two times will give us the total time for the cross-validation process.

Hold-out method time calculation: After finding the best λ value, we train the model again on the cross-validation data (800 observations) and test on the validation data (200 observations). The time for this step is calculated similarly as (800^2) for training and 0.5 * (200^2) for testing.

Adding the times: The total time for the entire procedure is the time for the cross-validation process plus the time for the hold-out method.

Doing these calculations step by step and understanding the purpose of each one helps to make the problem more manageable. It might be a complex problem, but breaking it down into smaller, understandable parts can make it much easier to tackle.
"""

"""
In machine learning, the process typically involves splitting the data into training, validation, and test sets. Here's how it works and why we're doing additional calculations after cross-validation:

Training set: This is used to train the model parameters. Here, we adjust the model's weights with the goal of minimizing the error on the training data.

Validation set: This is used to tune the hyperparameters of the model, like the regularization strength λ in this case. Cross-validation is a common method used for this purpose, especially when we don't have a lot of data. It splits the training data into K folds, trains the model on K-1 of those folds, and tests it on the remaining fold. It repeats this process K times so every fold serves as the test set once, and averages the results to get an estimate of the model's performance. In this problem, Alice performs cross-validation for different λ values on the cross-validation set, DCV (800 observations).

Test set: Once we've trained our model and chosen the best hyperparameters using the validation set, we want to estimate how well the model will perform on new, unseen data. This is the model's generalization error. The hold-out method is a common way to estimate this: we simply apply the model to the test set and see how well it performs. In this problem, Alice uses the hold-out method on the validation set, Dvalidation (200 observations), to estimate the generalization error.

That's why, after calculating the time for cross-validation, we calculate additional time for training on DCV and testing on Dvalidation. This gives us an estimate of the total computational cost, which includes not only finding the best λ but also estimating the final performance of the model.
"""

N = 1000
Dvalidation_ratio = 0.2
DCV_ratio = 1 - Dvalidation_ratio
K = 10 # number of folds
lambda_values = 6 # number of regularization strengths

# calculate sizes of the sets
Dvalidation = N * Dvalidation_ratio
DCV = N * DCV_ratio

# sizes for the cross-validation training and test sets
Ntrain = DCV * (K-1) / K
Ntest = DCV / K

# calculate training and testing time for the cross-validation
Ttrain = lambda_values * K * Ntrain**2
Ttest = lambda_values * K * 0.5 * Ntest**2

# training and testing on the full DCV and Dvalidation sets
Ttrain_final = DCV**2
Ttest_final = 0.5 * Dvalidation**2

# calculate total time
Ttotal = Ttrain + Ttest + Ttrain_final + Ttest_final
print(f"Total time: {Ttotal} units of time.")
print()
# 19

"""
This code standardizes the transformed datasets (by subtracting the column-wise mean and dividing by the column-wise standard deviation), then applies Ridge Regression to each. The Ridge regression fit is then done with alpha=0.25 (which corresponds to λ in your question). The learned parameters w0 (intercept) and w (weights) are printed out for each option.

Please note that in practice, Ridge Regression might not result in the exact weights given in your question due to its regularization nature, which discourages learning a model that's too complex. Thus, the results here will be approximations. The option whose weights are closest to [0.39, 0.77] will be the most likely transformation applied to the original data.
"""
"""
Option A: w = [0.0758, 0.437]
Option C: w = [0.356, 0.118]
Option D: w = [0.346, 0.682]

We can see that the weights produced by the transformations in Option D are closest to the weights given in the problem statement (w = [0.39, 0.77]). Therefore, it's likely that the transformation used was Option D: ˜xi = [ xi, x_i^2 ].
"""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# Original dataset
x = np.array([-0.5, 0.39, 1.19, -1.08]).reshape(-1, 1)
y = np.array([-0.86, -0.61, 1.37, 0.1])

# Transformed dataset for each option
x_transform_A = np.hstack([x, x**3])
x_transform_C = np.hstack([x, np.sin(x)])
x_transform_D = np.hstack([x, x**2])

# Standardize each transformed dataset
scaler = StandardScaler()
x_transform_A_standardized = scaler.fit_transform(x_transform_A)
x_transform_C_standardized = scaler.fit_transform(x_transform_C)
x_transform_D_standardized = scaler.fit_transform(x_transform_D)

# Perform Ridge Regression for each option
ridge = Ridge(alpha=0.25)

# Option A
ridge.fit(x_transform_A_standardized, y)
print('A: w0 = {}, w = {}'.format(ridge.intercept_, ridge.coef_))

# Option C
ridge.fit(x_transform_C_standardized, y)
print('C: w0 = {}, w = {}'.format(ridge.intercept_, ridge.coef_))

# Option D
ridge.fit(x_transform_D_standardized, y)
print('D: w0 = {}, w = {}'.format(ridge.intercept_, ridge.coef_))
print()

# 20
# Book ANN
# 15.2
"""
To determine the output of the neural network, let's follow the computation through the layers:

Input Layer:

x1 = 1
x2 = 2
Hidden Layer:

n1 = f(w31 * x1 + w32 * x2) = f(0.5 * 1 + (-0.4) * 2) = f(0.5 - 0.8) = f(-0.3) = 0 (thresholded linear activation)
n2 = f(w41 * x1 + w42 * x2) = f(0.4 * 1 + 0 * 2) = f(0.4) = 0.4 (thresholded linear activation)
Output Layer:

n3 = f(w53 * n1 + w54 * n2) = f((-0.4) * 0 + 0.1 * 0.4) = f(0.04) = 0.04 (thresholded linear activation)
Therefore, the output of the network is approximately:
ˆy = n3 = 0.04

The correct option is:
A) ˆy = 0.04
"""

# 21
# Book AUC
# 16.1
"""
For thresholds greater than 1, both TPR and FPR are 0. This forms the first point (0,0) of the ROC curve.

For a threshold of 1, FPR = 3/11 and TPR = 1. This forms the second point (3/11, 1) of the ROC curve. We can calculate the area of this trapezoid: 1/2 * base * (height1 + height2) = 1/2 * (3/11) * (1+0) = 3/22.

Lowering the threshold to 0, FPR = 1 and TPR = 1. This forms the third point (1,1) of the ROC curve. We calculate the area of this trapezoid as well: 1/2 * base * (height1 + height2) = 1/2 * (1-3/11) * (1+1) = 8/11.

Summing up the areas of the trapezoids: AUC = 3/22 + 8/11 = 0.864 (rounded to three decimal places).

Please note that the calculated AUC can vary slightly depending on the method of approximation and rounding used. In practice, AUC is often calculated using more complex datasets and algorithms for ROC curve generation and area calculation, and software packages often implement these calculations for us.
"""
# Define the points of the ROC curve
roc_points = [(0, 0), (3/11, 1), (1, 1)]

# Initialize AUC
auc = 0

# Calculate the area of each trapezoid and add to AUC
for i in range(1, len(roc_points)):
    # Calculate the base and the two heights of the trapezoid
    base = roc_points[i][0] - roc_points[i-1][0]
    height1 = roc_points[i-1][1]
    height2 = roc_points[i][1]

    # Calculate the area of the trapezoid and add it to the AUC
    auc += 0.5 * base * (height1 + height2)

# Print the AUC
print(auc)

# 16.2
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Actual class labels
y = [0, 0, 1, 1, 0, 1, 0]

# Predicted probabilities
probs = [0.15, 0.25, 0.4, 0.5, 0.55, 0.6, 0.61]

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y, probs)

# Compute ROC area
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# 16.3
# Given values
N = 1000  # Total number of observations
TN = 489  # Number of true negatives
FPR = 0.164  # False positive rate
TPR = 0.412  # True positive rate

# Calculate the total number of actual negative observations
N_negative = TN / (1 - FPR)

# Calculate the total number of actual positive observations
P = N - N_negative

# Calculate the number of true positives
TP = TPR * P

print("The number of true positives is approximately:", round(TP))
print()
# Given values
N = 1000  # Total number of observations
TN = 489  # Number of true negatives
FPR = 0.164  # False positive rate
TPR = 0.412  # True positive rate

# Calculate the number of false positives
FP = FPR * TN / (1 - FPR)

# Calculate the number of true positives
TP = TPR * (N - TN - FP)

print("The number of true positives is approximately:", round(TP))
print()





# 22
"""
To solve this problem, let's denote the support of X as supp(X), the support of Y as supp(Y), the confidence of the association rule X → Y as conf(X → Y), and the confidence of the association rule Y → X as conf(Y → X). We are given:

supp(X) = 3/5
supp(Y) = 8/15
conf(X → Y) = 1/6
We also know that the confidence of an association rule A → B can be calculated as:

conf(A → B) = supp(A ∩ B) / supp(A)

Let's first find the support of the intersection of X and Y, supp(X ∩ Y), using the given confidence of the rule X → Y:

supp(X ∩ Y) = conf(X → Y) * supp(X)
= (1/6) * (3/5)
= 3/30
= 1/10

Now, we can find the confidence of the association rule Y → X:

conf(Y → X) = supp(Y ∩ X) / supp(Y)
= supp(X ∩ Y) / supp(Y)
= (1/10) / (8/15)
= (1/10) * (15/8)
= 15/80
= 3/16

Therefore, the answer is:

D. conf(Y → X) = 3/16


-------------
Let's denote the support of X as sup(X), the support of Y as sup(Y), and the number of transactions containing both X and Y as sup(X, Y). We can express the confidence of X -> Y as:

scss
Copy code
conf(X -> Y) = sup(X, Y) / sup(X)
We know from the question that:

scss
Copy code
sup(X) = 3/5
sup(Y) = 8/15
conf(X -> Y) = 1/6
Let's first find sup(X, Y) from the confidence of X -> Y:

scss
Copy code
1/6 = sup(X, Y) / (3/5)
sup(X, Y) = 1/6 * 3/5
           = 3/30
           = 1/10
Now let's find the confidence of Y -> X, which is:

scss
Copy code
conf(Y -> X) = sup(X, Y) / sup(Y)
              = (1/10) / (8/15)
              = 15/80
              = 3/16
So, the answer is D. conf(Y -> X) = 3/16.

"""

# 23
# Book
import scipy.stats as stats

# Data from Table 11.2
# For fold 1
b1, c1 = 8, 7
# For fold 2
b2, c2 = 15, 11
# For fold 3
b3, c3 = 5, 17

# Summing them across the three folds
b = b1 + b2 + b3
c = c1 + c2 + c3

# Calculating the test statistic
chi_square = ((b - c) ** 2) / (b + c)

# Calculating the p-value using the exact binomial test.
# Note that we are using the tail probability.
p_value = stats.binom_test(min(b, c), n=b+c, p=0.5, alternative='two-sided')

print("Chi-Square Statistic:", chi_square)
print("P-Value:", p_value)

"""
In this question, we are asked to consider a statistical procedure to determine if there is a difference between the predictions of two regression models, M1 and M2, using a paired test. The distribution of the paired differences in predictions (yM1 - yM2) is shown as a histogram in Figure 9, which is not provided.

However, based on the information you've given, I can give you some insights on how to decide on the correct option.

The estimated difference ˆz should be some measure of central tendency of the differences in predictions (typically the mean of the differences).

The confidence interval (CI) should be an interval that you expect to contain the true mean difference with a certain level of confidence (typically 95%). It is centered around the estimated difference and accounts for the variability in the data.

The p-value should be a measure of the evidence against the null hypothesis that there is no difference between the predictions of M1 and M2. If it is less than the chosen significance level (typically 0.05), you might reject the null hypothesis. Importantly, p-values are always between 0 and 1, and they represent a probability, so a p-value less than -0.05 doesn't make sense.

Now, let's consider each option:

A. ˆz = 0.69, CI = [0.63, 0.75], p-value < 0.05

The CI is symmetric around ˆz, which makes sense.
The p-value is plausible (less than 0.05).
B. ˆz = −1.05, CI = [−1.29, −0.81], p-value < 0.05

The CI is symmetric around ˆz, which makes sense.
The p-value is plausible (less than 0.05).
C. ˆz = 0.63, CI = [0.57, 0.69], p-value < −0.05

The CI is symmetric around ˆz, which makes sense.
The p-value is not plausible (less than -0.05 doesn't make sense).
D. ˆz = 0.76, CI = [0.59, 0.76], p-value < 0.05

The CI is not symmetric around ˆz, which doesn't make sense unless there is a very good reason for it.
The p-value is plausible (less than 0.05).
Without seeing the histogram in Figure 9, based on the information provided, Option C can be ruled out because p-values can't be negative. Option D can probably be ruled out because usually, the CI would be symmetric around the estimated difference.

Options A and B both look plausible, but without the histogram, it's impossible to tell which one is correct as it depends on whether the differences in predictions are generally positive or negative.

Sure, let's work with the information you can extract from the plot. From the histogram showing the distribution of differences (yM1 - yM2) across the cross-validation folds, please try to identify the following information:

Central Tendency: Look for where the bulk of the data points lie. Is it around a positive value, negative value, or close to zero? If it's around a positive value, it suggests that M1 generally has higher predictions than M2, and vice versa.

Spread or Variation: Observe the width of the distribution. This will give you an idea about the variation in the differences.

Skewness: Is the histogram symmetric? If it's skewed to the left or right, it could influence the confidence interval.

Now, let's analyze the options again:

A. ˆz = 0.69, CI = [0.63, 0.75], p-value < 0.05

This option suggests that on average, M1 has higher predictions than M2 (positive ˆz). The CI is also positive, and the p-value indicates a statistically significant difference.
B. ˆz = −1.05, CI = [−1.29, −0.81], p-value < 0.05

This option suggests that on average, M2 has higher predictions than M1 (negative ˆz). The CI is also negative, and the p-value indicates a statistically significant difference.
C. ˆz = 0.63, CI = [0.57, 0.69], p-value < −0.05

We can rule this out because the p-value is negative, which is not possible.
D. ˆz = 0.76, CI = [0.59, 0.76], p-value < 0.05

The CI is not symmetric around ˆz, which is unusual.
Based on the observations from the histogram:

If the bulk of the differences are positive and there is not much skewness, Option A is more likely.
If the bulk of the differences are negative and there is not much skewness, Option B is more likely.
Please let me know if the information from the histogram aligns with either option A or B.


The bulk of the differences in the histogram can be determined by observing where most of the data is concentrated. Here's how you can interpret the histogram:

Positive Bulk: If most of the bars in the histogram are on the right side of the 0 (zero) mark, and there's a concentration of bars in the positive value range, then the bulk of the differences are positive. This would mean that, on average, the values of yM1 are greater than those of yM2.

Negative Bulk: If most of the bars in the histogram are on the left side of the 0 (zero) mark, and there's a concentration of bars in the negative value range, then the bulk of the differences are negative. This would mean that, on average, the values of yM2 are greater than those of yM1.

Around Zero: If the bars are centered around 0, this indicates that there isn't a consistent difference in either direction between yM1 and yM2.

Also, take note of the height of the bars. Higher bars indicate more data points within that range. So, if the bars on the positive side are significantly higher than on the negative side, it indicates a positive bulk, and vice versa.

The histogram provides a visual representation of the distribution of data. By looking at the shape and area where the bars are concentrated, you can infer the general behavior of the dataset regarding positive or negative differences.
"""


# 24
import math

# Given binarized dataset
dataset = {
    'o1': [1, 1, 0, 1, 0, 0, 'C1'],
    'o2': [1, 0, 1, 1, 0, 0, 'C1'],
    'o3': [0, 0, 1, 1, 1, 1, 'C1'],
    'o4': [0, 1, 1, 1, 1, 1, 'C1'],
    'o5': [1, 1, 0, 1, 1, 0, 'C2'],
    'o6': [1, 1, 0, 1, 1, 1, 'C2'],
    'o7': [0, 1, 1, 1, 0, 1, 'C2'],
    'o8': [0, 1, 0, 1, 1, 1, 'C2'],
    'o9': [0, 1, 0, 1, 1, 1, 'C2'],
    'o10': [1, 1, 1, 1, 1, 0, 'C2']
}

# Function representing the classifier f1
def f1(b1, b2, b3, b4, b5, b6):
    return 'C1' if b3 == 1 and b4 == 1 else 'C2'

# Calculate the weighted error ε1
num_samples = len(dataset)
sample_weight = 1 / num_samples
incorrect_classifications = 0

for observation, features in dataset.items():
    # Using f1 to classify the observation
    classification = f1(*features[:6])
    # Checking if the classification is incorrect
    if classification != features[6]:
        incorrect_classifications += 1

# Weighted error
epsilon_1 = incorrect_classifications * sample_weight

# Calculate the importance α1
alpha_1 = 0.5 * math.log((1 - epsilon_1) / epsilon_1)

# Output the result
print(f"The importance of the classifier f1, α1, is approximately {alpha_1:.2f}")


# 25
import scipy.stats as stats
import numpy as np

# Given component weights and covariance matrices
w = [0.5, 0.49, 0.01]
cov_matrices = [
    [[1.1, 2.0], [2.0, 5.5]],
    [[1.1, 0.0], [0.0, 5.5]],
    [[1.5, 0.0], [0.0, 1.5]]
]

# Sample component means for illustration (assuming non-zero means)
means = [[1.0, 1.0], [0.5, 0.5], [0.2, 0.2]]

# Compute the overall mean of the GMM
overall_mean = sum(w[k] * np.asarray(means[k]) for k in range(3))

# Compute the covariance matrix of the GMM
cov_gmm = sum(w[k] * (np.asarray(cov_matrices[k]) + np.outer(means[k], means[k])) for k in range(3)) - np.outer(overall_mean, overall_mean)

# Compute the eigenvectors of the covariance matrix
eigenvalues, eigenvectors = np.linalg.eig(cov_gmm)

# The principal component directions are the eigenvectors
# Sort them by eigenvalues
sorted_indices = np.argsort(eigenvalues)[::-1]
v1 = eigenvectors[:, sorted_indices[0]]
v2 = eigenvectors[:, sorted_indices[1]]

print("v1:", v1)
print("v2:", v2)


# 27
# Given data
X = np.array([-0.82, 0.0, 2.5])
N = len(X)
desired_L = np.array([-2.3, -2.3, -13.91])
lambda_values = [1.15, 0.15, 0.21, 0.49]

# Function to compute test log likelihood for a given λ
def compute_test_log_likelihood(X, N, lambda_value):
    L = np.zeros(N)
    for i in range(N):
        sum_gaussian_densities = 0
        for j in range(N):
            if j != i:
                # Using the Gaussian probability density function
                sum_gaussian_densities += stats.norm.pdf(X[i], loc=X[j], scale=lambda_value)
        # Compute the log density
        L[i] = np.log(sum_gaussian_densities / (N - 1))
    return L

# Check which λ gives a test log likelihood close to the desired values
for lambda_value in lambda_values:
    L = compute_test_log_likelihood(X, N, lambda_value)
    print(f"λ = {lambda_value}, L = {L}")

# Note: Due to numerical precision, you might need to check if the computed L values
# are approximately equal to the desired values.
