
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
