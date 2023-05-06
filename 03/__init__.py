######################################
# 03
######################################
print("Section 03:")
# 1
"""
Now, let's discuss the given options:

A. To check if relatively high values of CAL, PROT, FAT, FIB, SUG, POT, VIT, SHELF, and WEIGHT and low values of TYPE, SOD, CARB, and CUPS will result in a negative projection onto the first principal component, we can examine the signs of the values in v1.

B. To check if PCA2 primarily discriminates between relatively low values of PROT and high values of SHELF, we can examine the values corresponding to PROT and SHELF in v2.

C. Without actual data, we cannot directly check the correlation between the data projected onto the second principal component (i.e., PCA2) and RAT.

D. The principal component directions v1 and v2 are guaranteed to be orthogonal to each other since the data has been standardized and PCA is performed.
"""
import numpy as np
v1 = np.array([0.0247, -0.0388, -0.3288, -0.2131, 0.0477, -0.4584, 0.2683, -0.0838, -0.5020, -0.0200, -0.3091, -0.2588, 0.3714])
v2 = np.array([-0.0764, 0.5675, -0.0550, 0.2449, 0.3115, -0.1999, 0.1738, 0.3668, -0.0737, 0.2988, 0.0628, 0.4446, 0.1051])

# 2
"""
A is incorrect because to calculate the proportion of variation explained by the first three principal components, you need to square the singular values and sum them up: σ1^2 + σ2^2 + σ3^2. Then divide this sum by the total sum of squared singular values: (σ1^2 + σ2^2 + σ3^2) / (σ1^2 + σ2^2 + σ3^2 + σ4^2 + σ5^2 + σ6^2). This value is less than 0.9, so less than 90% of the variation in the data is accounted for by the first three principal components.

B is correct because the positive coefficients in v1 are associated with Age, MW, and HT. This means that relatively heavy, old, and white mothers that frequently go to the physician and have a history of hypertension but do not have uterine irritability will have a positive projection onto the first principal component.

C is correct because the positive coefficients in v2 are associated with MW, Race, and HT, while the negative coefficient is associated with PV. This means that relatively young, heavy mothers that are not white and have a history of hypertension but infrequently go to the physician and do not have uterine irritability will have a positive projection onto the second principal component.

D is correct because since the data is standardized, the mean is already subtracted, and you can directly carry out the singular value decomposition on the standardized data. Standardization ensures that the data is centered around the origin, which is a necessary step in PCA to ensure that the first principal component captures the direction of maximum variance.
"""
import numpy as np

# Singular values
singular_values = np.array([17.0, 15.2, 13.1, 13.0, 11.8, 11.3])

# Squared singular values
squared_singular_values = singular_values ** 2

# Proportion of variation explained by the first three principal components
first_three_pcs_var = np.sum(squared_singular_values[:3]) / np.sum(squared_singular_values)

print('3.2')
print("Proportion of variation explained by the first three principal components:", first_three_pcs_var)

# Check if more than 90% of the variation is accounted for by the first three principal components
print("More than 90% of variation is accounted for by the first three principal components:", first_three_pcs_var > 0.9)
print('\n')

# 3
"""

"""


######################################
# 04
######################################
print("Section 04:")
# 1
"""
The correct answer is C: To compare the similarity of observations o1 and o3, we can use the Simple Matching Coefficient (SMC). The SMC is a similarity measure for binary data, which is suitable for the given dataset. It is calculated as:

SMC(o1, o3) = (number of matching attributes) / (number of total attributes)

Given that the cityblock distance between o1 and o3 is 7, it means that there are 7 attributes in which o1 and o3 differ, and 8 attributes (15 - 7) in which they match. Thus, the SMC for o1 and o3 is:

SMC(o1, o3) = 8 / 15 ≈ 0.533

So, the similarity between observation o1 and o3 using the Simple Matching Coefficient is approximately 0.533.

Here's why the other options are incorrect:

A: COS(o1, o3) = 0.533
The cosine similarity (COS) is used for continuous data and is not a suitable measure for binary data. In this case, the data consists of binary vectors, so the cosine similarity is not appropriate to determine the similarity between o1 and o3.

B: J(o1, o3) = 0.533
The Jaccard coefficient (J) is a measure of similarity between two sets, where it calculates the ratio of the size of the intersection of the sets to the size of the union of the sets. In this case, we have binary vectors, not sets. Thus, the Jaccard coefficient is not the right choice for determining the similarity between o1 and o3.

D: There is insufficient information to draw specific conclusions.
Although the given information may seem limited, it is enough to calculate the Simple Matching Coefficient (SMC) between o1 and o3. Using the cityblock distance, we can find the number of matching and non-matching attributes, which is sufficient for calculating the SMC.
"""
cityblock_distance_o1_o3 = 7
total_attributes = 15

# Calculate the number of matching attributes
matching_attributes = total_attributes - cityblock_distance_o1_o3

# Calculate the Simple Matching Coefficient (SMC)
smc_o1_o3 = matching_attributes / total_attributes
print('4.1')
print(f'smc_o1_o3: {smc_o1_o3}')
print('\n')

# 2
"""
The correct answer is D. The three measures of similarity are given by:

Jaccard Coefficient (J): J(x, y) = f11 / (f01 + f10 + f11)
Simple Matching Coefficient (SMC): SMC(x, y) = (f00 + f11) / (f01 + f10 + f11 + f00)
Cosine Similarity (cos): cos(x, y) = (x · y) / (∥x∥∥y∥)


Sure, let's calculate the Jaccard Coefficient, Simple Matching Coefficient, and Cosine Similarity for the given pairs:

1-
J(NS1, NS2) and SMC(NS1, NS2)
NS1: 1 0 0 1 0 1 1 0
NS2: 0 1 1 0 1 0 1 0
f11: 2 (indices 6 and 7)
f01: 3 (indices 1, 3, and 5)
f10: 3 (indices 2, 4, and 5)
f00: 0 (no matching 0's)

J(NS1, NS2) = f11 / (f01 + f10 + f11) = 2 / (3 + 3 + 2) = 1/4
SMC(NS1, NS2) = (f00 + f11) / (f01 + f10 + f11 + f00) = (0 + 2) / (3 + 3 + 2 + 0) = 2/8 = 1/4

2- 
cos(NS4, NS5)
NS4: 0 1 1 0 0 1 1 0
NS5: 1 0 1 0 1 0 1 0
x · y = (01 + 10 + 11 + 00 + 01 + 10 + 11 + 00) = 2
∥x∥ = sqrt(0^2 + 1^2 + 1^2 + 0^2 + 0^2 + 1^2 + 1^2 + 0^2) = sqrt(4) = 2
∥y∥ = sqrt(1^2 + 0^2 + 1^2 + 0^2 + 1^2 + 0^2 + 1^2 + 0^2) = sqrt(4) = 2

cos(NS4, NS5) = (x · y) / (∥x∥∥y∥) = 2 / (2 * 2) = 1/2

3-
SMC(NS5, AS5) and J(NS5, AS5)
NS5: 1 0 1 0 1 0 1 0
AS5: 1 0 1 0 0 1 1 0
f11: 4 (indices 1, 3, 6, and 7)
f01: 1 (index 5)
f10: 1 (index 5)
f00: 2 (indices 2 and 4)

SMC(NS5, AS5) = (f00 + f11) / (f01 + f10 + f11 + f00) = (2 + 4) / (1 + 1 + 4 + 2) = 6/8 = 3/4
J(NS5, AS5) = f11 / (f01 + f10 + f11) = 4 / (1 + 1 + 4) = 4/6 = 2/3

Finally, we find the correct answer to be D: cos(NS5, AS5) = 3/4.
"""
import numpy as np
from scipy.spatial.distance import jaccard, cosine

def simple_matching_coefficient(a, b):
    return np.sum(a == b) / len(a)

NS1 = np.array([1, 0, 0, 1, 0, 1, 1, 0])
NS2 = np.array([0, 1, 1, 0, 1, 0, 1, 0])
NS4 = np.array([0, 1, 1, 0, 0, 1, 1, 0])
NS5 = np.array([1, 0, 1, 0, 1, 0, 1, 0])
AS5 = np.array([1, 0, 1, 0, 0, 1, 1, 0])

# Calculate the Jaccard Coefficient and Simple Matching Coefficient for NS1 and NS2
J_NS1_NS2 = jaccard(NS1, NS2)
SMC_NS1_NS2 = simple_matching_coefficient(NS1, NS2)

# Calculate the Cosine Similarity for NS4 and NS5
cos_NS4_NS5 = 1 - cosine(NS4, NS5)

# Calculate the Jaccard Coefficient and Simple Matching Coefficient for NS5 and AS5
J_NS5_AS5 = jaccard(NS5, AS5)
SMC_NS5_AS5 = simple_matching_coefficient(NS5, AS5)

# Calculate the Cosine Similarity for NS5 and AS5
cos_NS5_AS5 = 1 - cosine(NS5, AS5)

print('4.2')
print("J(NS1, NS2):", J_NS1_NS2)
print("SMC(NS1, NS2):", SMC_NS1_NS2)
print("cos(NS4, NS5):", cos_NS4_NS5)
print("J(NS5, AS5):", J_NS5_AS5)
print("SMC(NS5, AS5):", SMC_NS5_AS5)
print("cos(NS5, AS5):", cos_NS5_AS5)
print('\n')

# 3
"""
Let's calculate the Jaccard Coefficient, Simple Matching Coefficient, and Cosine Similarity for the given pairs:

1-
J(S1, S2), J(S1, NS1)
S1: 1 0 1 0 1 0
S2: 1 0 1 0 0 1
NS1: 0 1 1 0 1 0
f11(S1, S2): 3 (indices 1, 2, and 3)
f01(S1, S2): 1 (index 5)
f10(S1, S2): 1 (index 6)

f11(S1, NS1): 2 (indices 2 and 5)
f01(S1, NS1): 2 (indices 1 and 3)
f10(S1, NS1): 1 (index 4)

J(S1, S2) = f11(S1, S2) / (f01(S1, S2) + f10(S1, S2) + f11(S1, S2)) = 3 / (3 + 1 + 1) = 3/5
J(S1, NS1) = f11(S1, NS1) / (f01(S1, NS1) + f10(S1, NS1) + f11(S1, NS1)) = 2 / (2 + 2 + 1) = 2/5

2-
SMC(S1, S2), SMC(S1, NS1)
f00(S1, S2): 1 (index 4)
f00(S1, NS1): 1 (index 6)
SMC(S1, S2) = (f00(S1, S2) + f11(S1, S2)) / (f01(S1, S2) + f10(S1, S2) + f11(S1, S2) + f00(S1, S2)) = (1 + 3) / (1 + 1 + 3 + 1) = 4/6 = 2/3
SMC(S1, NS1) = (f00(S1, NS1) + f11(S1, NS1)) / (f01(S1, NS1) + f10(S1, NS1) + f11(S1, NS1) + f00(S1, NS1)) = (1 + 2) / (2 + 1 + 2 + 1) = 3/6 = 1/2

3-
cos(S1, S2), cos(S1, NS1)
x · y(S1, S2) = (11 + 00 + 11 + 00 + 10 + 01) = 3
x · y(S1, NS1) = (10 + 01 + 11 + 00 + 11 + 00) = 2
∥S1∥ = sqrt(1^2 + 0^2 + 1^2 + 0^2 + 1^2 + 0^2) = sqrt(3)
∥S2∥ = sqrt(1^2 + 0^2 + 1^2 + 0^2 + 0^2 + 1^2) = sqrt(3)
∥NS1∥ = sqrt(0^2 + 1^2 + 1^2 + 0^2 + 1^2 + 0^2) = sqrt(3)

cos(S1, S2) = (x · y(S1, S2)) / (∥S1∥∥S2∥) = 3 / (sqrt(3) * sqrt(3)) = 3/3 = 1
cos(S1, NS1) = (x · y(S1, NS1)) / (∥S1∥∥NS1∥) = 2 / (sqrt(3) * sqrt(3)) = 2/3

Now let's compare the options:

A. J(S1, S2) > J(S1, NS1) -> 3/5 > 2/5 -> True
B. SMC(S1, S2) > SMC(S1, NS1) -> 2/3 > 1/2 -> True
C. J(S1, S2) = cos(S1, S2) -> 3/5 = 1 -> False
D. SMC(S1, S2) = cos(S1, S2) -> 2/3 = 1 -> False

The correct answer is A. Using the Jaccard coefficient, S1 is more similar to S2 than to NS1, i.e., J(S1, S2) > J(S1, NS1).
"""

from math import sqrt

def jaccard(x, y):
    intersection = sum([a == 1 and b == 1 for a, b in zip(x, y)])
    union = sum([a == 1 or b == 1 for a, b in zip(x, y)])
    return intersection / union

def smc(x, y):
    matches = sum([a == b for a, b in zip(x, y)])
    return matches / len(x)

def cosine_similarity(x, y):
    dot_product = sum([a * b for a, b in zip(x, y)])
    x_norm = sqrt(sum([a * a for a in x]))
    y_norm = sqrt(sum([a * a for a in y]))
    return dot_product / (x_norm * y_norm)

S1 = [1, 0, 1, 0, 1, 0]
S2 = [1, 0, 1, 0, 0, 1]
NS1 = [0, 1, 1, 0, 1, 0]

j_s1_s2 = jaccard(S1, S2)
j_s1_ns1 = jaccard(S1, NS1)

smc_s1_s2 = smc(S1, S2)
smc_s1_ns1 = smc(S1, NS1)

cos_s1_s2 = cosine_similarity(S1, S2)
cos_s1_ns1 = cosine_similarity(S1, NS1)

j_s1_s2_rounded = round(j_s1_s2, 2)
j_s1_ns1_rounded = round(j_s1_ns1, 2)

smc_s1_s2_rounded = round(smc_s1_s2, 2)
smc_s1_ns1_rounded = round(smc_s1_ns1, 2)

cos_s1_s2_rounded = round(cos_s1_s2, 2)
cos_s1_ns1_rounded = round(cos_s1_ns1, 2)

print('4.3')
print("J(S1, S2) =", j_s1_s2)
print("J(S1, NS1) =", j_s1_ns1)
print("SMC(S1, S2) =", smc_s1_s2)
print("SMC(S1, NS1) =", smc_s1_ns1)
print("cos(S1, S2) =", cos_s1_s2)
print("cos(S1, NS1) =", cos_s1_ns1)

condition1 = j_s1_s2_rounded > j_s1_ns1_rounded
condition2 = smc_s1_s2_rounded > smc_s1_ns1_rounded
condition3 = j_s1_s2_rounded == cos_s1_s2_rounded
condition4 = smc_s1_s2_rounded == cos_s1_s2_rounded

print("J(S1, S2) > J(S1, NS1):", condition1)
print("SMC(S1, S2) > SMC(S1, NS1):", condition2)
print("J(S1, S2) == cos(S1, S2):", condition3)
print("SMC(S1, S2) == cos(S1, S2):", condition4)
print('\n')

# 4
"""
 I can help you match the histograms to the attributes based on the summary statistics provided.

When matching histograms to the attributes, consider the mean, standard deviation, and percentiles:

Look for a histogram with a mean close to the provided mean and a shape that reflects the standard deviation.
Check the percentiles (25%, 50%, and 75%) and see if they align with the distribution in the histogram.
Go through each attribute and compare its summary statistics to the histograms. Find the one that best matches the given statistics.

Remember that I cannot see the histograms, so you will need to do the visual comparison yourself.


 The correct answer is D: To solve the problem, note that we can read of the median,
25’th, and 75’th percentiles from table 4.5 as qp=50%, qp=25%, and qp=75% respectively. These can be
matched to the histograms in fig. 4.4 by observing histogram 2 does not have observations above 25
and thus must therefore be DeathRt. Histogram 4 is the only histogram having observations above
88.25 which only holds for InfMort (see 75th percentile). This only holds for answer option D.
"""
import numpy as np
import matplotlib.pyplot as plt

# Set the random seed for reproducibility
np.random.seed(42)

# Define the summary statistics
summary_stats = {
    "BirthRt": {"mean": 29.46, "std": 13.62, "n": 100},
    "DeathRt": {"mean": 10.73, "std": 4.66, "n": 100},
    "InfMort": {"mean": 55.28, "std": 46.05, "n": 100},
    "LExpM": {"mean": 61.38, "std": 9.67, "n": 100},
}

# Generate random data that matches the summary statistics
datasets = {}
for key, stats in summary_stats.items():
    data = np.random.normal(stats["mean"], stats["std"], stats["n"])
    datasets[key] = data

# Plot the histograms
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

axs[0, 0].hist(datasets["BirthRt"], bins=20)
axs[0, 0].set_title("BirthRt 4.4")

axs[0, 1].hist(datasets["DeathRt"], bins=20)
axs[0, 1].set_title("DeathRt")

axs[1, 0].hist(datasets["InfMort"], bins=20)
axs[1, 0].set_title("InfMort")

axs[1, 1].hist(datasets["LExpM"], bins=20)
axs[1, 1].set_title("LExpM")
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Set the random seed for reproducibility
np.random.seed(42)

# Define the summary statistics
summary_stats = {
    "BirthRt": {"mean": 29.46, "std": 13.62, "p25": 14.6, "p50": 29, "p75": 42.575, "n": 100},
    "DeathRt": {"mean": 10.73, "std": 4.66, "p25": 7.7, "p50": 9.5, "p75": 12.4, "n": 100},
    "InfMort": {"mean": 55.28, "std": 46.05, "p25": 13.025, "p50": 43, "p75": 88.25, "n": 100},
    "LExpM": {"mean": 61.38, "std": 9.67, "p25": 55.2, "p50": 63.4, "p75": 68.55, "n": 100},
}

# Generate random data that approximates the summary statistics
datasets = {}
for key, stats in summary_stats.items():
    data = np.random.normal(stats["mean"], stats["std"], stats["n"])
    data = np.sort(data)
    p25, p50, p75 = np.percentile(data, [25, 50, 75])

    scale_factor = (stats["p75"] - stats["p25"]) / (p75 - p25)
    shift_factor = stats["p50"] - p50 * scale_factor

    data = data * scale_factor + shift_factor
    datasets[key] = data

# Plot the histograms
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

axs[0, 0].hist(datasets["BirthRt"], bins=20)
axs[0, 0].set_title("BirthRt 4.4")

axs[0, 1].hist(datasets["DeathRt"], bins=20)
axs[0, 1].set_title("DeathRt")

axs[1, 0].hist(datasets["InfMort"], bins=20)
axs[1, 0].set_title("InfMort")

axs[1, 1].hist(datasets["LExpM"], bins=20)
axs[1, 1].set_title("LExpM")

plt.show()

# 4.5
"""
The correct answer for Question 5 is A.

A) The Mahanalobis distance between the two green circles is smaller than the Mahanalobis distance between the two black squares.

This is because the green circles are closer to each other in the direction of the positive correlation between the variables x1 and x2. The covariance matrix is positively correlated, and therefore, the Mahalanobis distance between the two green circles is smaller than between the two black squares.

The other options are incorrect for the following reasons:

B) The Mahanalobis distance between the two red crosses is the same as the Mahanalobis distance between the two green circles.

This is incorrect because the red crosses and green circles are not in the same direction concerning the covariance matrix, and their distances will not be the same.

C) The Mahanalobis distance between the two black squares is smaller than the Mahanalobis distance between the two cyan plusses.

This is incorrect because the black squares are farther apart from each other in the direction of the covariance matrix than the cyan plusses, so their Mahalanobis distance will be larger, not smaller.

D) The empirical covariance matrix estimated based on the blue observations has at least one element that is negative.

This is incorrect because, as mentioned before, the correlation between x1 and x2 is positive, and therefore the covariance matrix will only have positive elements.
"""
import numpy as np
from scipy.spatial import distance

# Replace these with the actual mean and covariance matrix calculated from the dataset
mean = np.array([0, 0])
cov_matrix = np.array([[1, 0], [0, 1]])

red_crosses = np.array([(-0.1, 0.1), (0.1, -0.1)])
green_circles = np.array([(-0.1, -0.1), (0.1, 0.1)])
cyan_plusses = np.array([(-0.12, -0.12), (0.12, 0.12)])
black_squares = np.array([(-0.2, -0.1), (0.2, 0.1)])

m_dist_red_crosses = distance.mahalanobis(red_crosses[0], red_crosses[1], np.linalg.inv(cov_matrix))
m_dist_green_circles = distance.mahalanobis(green_circles[0], green_circles[1], np.linalg.inv(cov_matrix))
m_dist_cyan_plusses = distance.mahalanobis(cyan_plusses[0], cyan_plusses[1], np.linalg.inv(cov_matrix))
m_dist_black_squares = distance.mahalanobis(black_squares[0], black_squares[1], np.linalg.inv(cov_matrix))

print('4.5')
print("Mahalanobis distance between red crosses:", m_dist_red_crosses)
print("Mahalanobis distance between green circles:", m_dist_green_circles)
print("Mahalanobis distance between cyan plusses:", m_dist_cyan_plusses)
print("Mahalanobis distance between black squares:", m_dist_black_squares)
print('\n')
######################################
# 05
######################################
print("Section 05:")
"""
We can use Bayes' theorem to calculate the probability that a random image contains a penguin given that it is classified as a penguin. Bayes' theorem states that:

P(A|B) = P(B|A) * P(A) / P(B)

In our case:

A: The image contains a penguin
B: The image is classified as a penguin

We are given the following probabilities:

P(A) = 0.01 (1% of images contain a penguin)
P(B|A) = 0.97 (network classifies an image as a penguin correctly with a 97% probability)
P(B|not A) = 0.03 (network classifies an image as a penguin incorrectly with a 3% probability)

First, we need to find P(B), the probability of an image being classified as a penguin:

P(B) = P(B|A) * P(A) + P(B|not A) * P(not A)

P(not A) is the probability that an image does not contain a penguin, which is 1 - P(A) = 0.99.

Now, we can calculate P(B):

P(B) = 0.97 * 0.01 + 0.03 * 0.99

Finally, we can apply Bayes' theorem to find P(A|B):

P(A|B) = P(B|A) * P(A) / P(B)

This will give us the probability that a random image contains a penguin given that it is classified as a penguin.

Sure, let's calculate the final answer using the probabilities we've discussed.

P(A) = 0.01
P(B|A) = 0.97
P(B|not A) = 0.03
P(not A) = 0.99

First, we calculate P(B):

P(B) = P(B|A) * P(A) + P(B|not A) * P(not A)
P(B) = 0.97 * 0.01 + 0.03 * 0.99
P(B) ≈ 0.0394

Now, we can apply Bayes' theorem to find P(A|B):

P(A|B) = P(B|A) * P(A) / P(B)
P(A|B) = 0.97 * 0.01 / 0.0394
P(A|B) ≈ 0.2462

So, the probability that a random image contains a penguin given that it is classified as a penguin is approximately 24.62%.
"""
P_A = 0.01
P_B_given_A = 0.97
P_B_given_not_A = 0.03
P_not_A = 0.99

P_B = P_B_given_A * P_A + P_B_given_not_A * P_not_A
P_A_given_B = P_B_given_A * P_A / P_B

print('5.1')
print("The probability that a random image contains a penguin given that it is classified as a penguin:", P_A_given_B)
print('\n')

######################################
# 06
######################################
print("Section 06:")
# 1
"""
The correct answer is C
Let's denote the events as follows:

R: the car is red
B: the car is blue
H: the car is heavy
L: the car is light
F: the car has four doors
T: the car has two doors
From the given information, we can find the following probabilities:

P(F) = 0.5 (probability of having four doors)
P(H|F) = 0.8 (probability of being heavy given it has four doors)
P(H|T) = 0.2 (probability of being heavy given it has two doors)
P(H ∩ R) = 0.1 (probability of being heavy and red)
We are asked to find P(B|H), the probability that a car is blue given it is heavy.

First, let's find P(H). We know that P(F) + P(T) = 1. We can find P(H) using the law of total probability:

P(H) = P(H|F) * P(F) + P(H|T) * P(T) = 0.8 * 0.5 + 0.2 * 0.5 = 0.5

Now, let's find P(B ∩ H). Since a car can only be red or blue, we have P(B) + P(R) = 1, and P(H ∩ B) = P(H) - P(H ∩ R):

P(B ∩ H) = P(H) - P(H ∩ R) = 0.5 - 0.1 = 0.4

Finally, we can use the definition of conditional probability to find P(B|H):

P(B|H) = P(B ∩ H) / P(H) = 0.4 / 0.5 = 0.8

So, the probability that a car is blue given it is heavy is 0.8.

"""
# Probabilities given in the problem
P_F = 0.5
P_H_given_F = 0.8
P_H_given_T = 0.2
P_H_and_R = 0.1

# Calculate P(H)
P_H = P_H_given_F * P_F + P_H_given_T * (1 - P_F)

# Calculate P(B and H)
P_B_and_H = P_H - P_H_and_R

# Calculate P(B|H)
P_B_given_H = P_B_and_H / P_H

print('6.1')
print(f'P_B_given_H: {P_B_given_H}')
print('\n')

# 2
"""
The correct answer is D
To find the probability that a subject who has survived would have positive axillary nodes, we can use the conditional probability formula. Let's denote the events as follows:

P(PAN) = Probability of positive axillary nodes
P(Surv) = Probability of survival after 5 years
P(PAN|Surv) = Probability of positive axillary nodes given survival
From the problem, we are given:

P(PAN) = 0.56
P(Surv|PAN) = 0.36
P(Surv|not PAN) = 0.14
We want to find P(PAN|Surv). Using Bayes' theorem, we can write:

P(PAN|Surv) = P(Surv|PAN) * P(PAN) / P(Surv)

We need to find P(Surv) first. To do that, we can use the law of total probability:

P(Surv) = P(Surv|PAN) * P(PAN) + P(Surv|not PAN) * P(not PAN)
"""
P_PAN = 0.56
P_Surv_given_PAN = 0.36
P_Surv_given_not_PAN = 0.14

P_not_PAN = 1 - P_PAN

# Calculate P(Surv)
P_Surv = P_Surv_given_PAN * P_PAN + P_Surv_given_not_PAN * P_not_PAN

# Calculate P(PAN|Surv)
P_PAN_given_Surv = P_Surv_given_PAN * P_PAN / P_Surv

print('6.2')
print(f'P_PAN_given_Surv: {P_PAN_given_Surv}')
print('\n')

# 3
"""
In this question, we are asked to find the expected value (E[x]) of a random variable x, given its probability density function (p(x)). The expected value can be thought of as the average or mean value of the random variable over many realizations. It is a measure of the central tendency of the distribution.

In the given probability density function, there are three linear segments that define the density function:

From 0 to 0.2, the density function is p(x) = 0.6 * x.
From 0.2 to 0.6, the density function is p(x) = 1 * x.
From 0.6 to 0.9, the density function is p(x) = 1.6 * x.
To find the expected value, we need to integrate the product of x and p(x) over the entire range of the random variable x. In this case, the range is from 0 to 0.9. The integral of xp(x) for each segment is calculated and summed up.

For segment 1 (0 to 0.2):
E[x]_1 = ∫(x * 0.6 * x) dx = 0.6 * ∫(x^2) dx

For segment 2 (0.2 to 0.6):
E[x]_2 = ∫(x * 1 * x) dx = ∫(x^2) dx

For segment 3 (0.6 to 0.9):
E[x]_3 = ∫(x * 1.6 * x) dx = 1.6 * ∫(x^2) dx

Now, we need to find the definite integral of x^2 for each segment, which is given by (1/3)(b^3 - a^3), where a and b are the limits of integration.

Finally, we sum up the expected values for each segment to get the overall expected value E[x]:

E[x] = E[x]_1 + E[x]_2 + E[x]_3

By calculating these integrals and summing them up, we obtain the expected value E[x] = 0.532.
"""
# Calculate the expected value E[x] for the given density function
E_x = 0.6 * 0.5 * (0.2**2 - 0**2) + 1 * 0.5 * (0.6**2 - 0.2**2) + 1.6 * 0.5 * (0.9**2 - 0.6**2)
print('6.3')
print(f'E_x: {E_x}')
print('\n')



######################################
# 07
######################################
print("Section 07:")
# 1
"""
The revised answer is:

7.1 The correct answer is B:

A: From Figure 7.16, we can see that the 25th and 50th percentiles (the median) of the attribute DB coincide, but the 50th and 75th percentiles do not.

B: Even though the distribution of AlA and AsA may have a similar shape, this does not imply that the two attributes are correlated. Correlation measures the strength and direction of the linear relationship between two variables, while the distribution shape only reflects the overall structure of the data for each variable separately. Therefore, this statement is correct.

C: The attribute TB is not likely to be normally distributed, as this attribute has a highly right-skewed distribution, which can be seen in the boxplot.

D: The attribute GDR does not have a clear outlier. In fact, the outlier corresponds to the females in the dataset, and all we can deduce from the plot is that more than 75% of the observations are males.

So, the correct statement is B.

"""

import seaborn as sns
import matplotlib.pyplot as plt

# Load a sample dataset from Seaborn
data = sns.load_dataset('iris')

# Standardize the data
data_standardized = (data.iloc[:, :4] - data.iloc[:, :4].mean()) / data.iloc[:, :4].std()

# Create a boxplot of the standardized data
sns.boxplot(data=data_standardized)
plt.title("Boxplots of Standardized Attributes 7.1")
plt.show()

# 2
"""
To identify the boxplot that represents the dataset, we can analyze the quartiles:

Given the dataset:

40 observations with the value 1
10 observations with the value 2
10 observations with the value 3
Total number of observations (N) = 60.

We need to find the quartiles (Q1, Q2, and Q3) of this dataset.

Q1 (25th percentile): 25% of the data is below this value. Since 25% of 60 is 15, and there are 40 observations with the value 1, the 25th percentile (Q1) is 1.

Q2 (50th percentile or median): 50% of the data is below this value. Since 50% of 60 is 30, and there are 40 observations with the value 1, the 50th percentile (Q2 or median) is also 1.

Q3 (75th percentile): 75% of the data is below this value. Since 75% of 60 is 45, we have 40 observations with the value 1 and 10 observations with the value 2, which together make up 50 observations. Therefore, the 75th percentile (Q3) is 2.

Now, we have the quartiles:

Q1: 1
Q2: 1
Q3: 2

25th percentile (Q1): 25% of the data is below this value.
50th percentile (Q2 or median): 50% of the data is below this value.
75th percentile (Q3): 75% of the data is below this value.

Let's analyze the dataset:
40 out of 60 observations have the value 1. That's 66.7% of the data.
50 out of 60 observations have the value 1 or 2. That's 83.3% of the data.
All 60 observations have the value 1, 2, or 3. That's 100% of the data.
From this analysis, we can deduce the following quartiles:

Q1 = 1 (since 66.7% > 25%)
Q2 = 1 (since 66.7% > 50%)
Q3 = 2 (since 83.3% > 75%)
Boxplot A has the quartiles that match the analysis, so it's the correct representation of the dataset.

- but why every thing below median is conside with it 

I understand why this may be confusing. In this particular dataset, the majority of the observations have the same value (1), which contributes to the quartiles having the same or very close values.

Let's break it down further:

40 out of 60 observations have the value 1. That's roughly 67% of the dataset.
10 out of 60 observations have the value 2. That's roughly 17% of the dataset.
10 out of 60 observations have the value 3. That's roughly 17% of the dataset.
When calculating the quartiles, we are trying to find the values that divide the dataset into four equal parts. Since the majority of the dataset (67%) has the same value (1), the first two quartiles (Q1 and Q2) will be the same, as more than 50% of the dataset consists of that value.

The third quartile (Q3) is the value that separates the lowest 75% of the data from the highest 25%. In this case, 40 observations have the value 1 and 10 observations have the value 2, which together make up 50 out of 60 observations (roughly 83% of the dataset). So, the 75th percentile (Q3) is 2.

In summary, when there is a high concentration of the same value in a dataset, it is possible for multiple quartiles to coincide with that value. In this example, since 67% of the dataset has the value 1, both Q1 and Q2 end up being 1.
"""
import numpy as np
import matplotlib.pyplot as plt

# Create the dataset
data = [1]*40 + [2]*10 + [3]*10
data = np.array(data)

# Calculate quartiles
q1 = np.percentile(data, 25)
q2 = np.percentile(data, 50)
q3 = np.percentile(data, 75)

print(f"Q1: {q1}, Q2: {q2}, Q3: {q3}")

# Plot the boxplot
plt.boxplot(data)
plt.title("Boxplot A 7.2")
plt.show()

# 3
"""
The correct answer is D: The mean and median values are not likely to be very close to each other for any of the attributes.

Let's go through the options to see why the others are not correct:

A) The boxplot does contain some outliers, but it's not necessary to remove them without further analysis. Outliers could be genuine data points representing extreme cases or they could be errors. It's essential to investigate them further before deciding to remove them or not.

B) None of the attributes appear to be normally distributed. Most of them are right-skewed, as shown by the longer tails on the right side of the boxplots.

C) If we don't standardize the data, a PCA would actually give more importance to the attributes with larger variance. Standardizing the data ensures that all attributes contribute equally to the PCA.

D) The mean and median values are not likely to be very close to each other for any of the attributes because the distributions are skewed. In skewed distributions, the mean is pulled towards the tail, making it different from the median, which is the middle value of the distribution.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew

# Load the dataset (assuming it's saved as 'wholesale_customers.csv')
data = pd.read_csv('wholesale_customers.csv')

# Create boxplots for each attribute
plt.figure(figsize=(12, 6))
sns.boxplot(data=data)
plt.xticks(rotation=45)
plt.title("Boxplot of the Six Input Attributes")
plt.show()

# Check the skewness of each attribute
print("Skewness of the attributes:")
print(data.skew())
"""


######################################
# 08
######################################
print("Section 08:")
# 1
"""
The linear regression model given is:

f(x) = 3.27 + 0.36x1 - 0.01x2 - 0.19x3 + 0.01x4 + 0.05x7

Let's analyze each statement:

A. According to the model, WINDDIR and HOUR are not relevant for predicting the pollution level.

This statement is correct. The model does not include coefficients for x5 (WINDDIR) and x6 (HOUR), indicating that they are not relevant for predicting the pollution level.

B. According to the model, fewer cars and more wind will result in lower pollution levels.

This statement is correct. The model has a positive coefficient for x1 (logCAR) and a negative coefficient for x3 (WIND). A decrease in x1 and an increase in x3 would result in a decrease in the pollution level.

The coefficient for x1 (logCAR) is 0.36, and for x3 (WIND) is -0.19.

Positive coefficients: When the coefficient of an input attribute is positive, it means that an increase in the value of that attribute will result in an increase in the output value. In this case, the positive coefficient for x1 (logCAR) indicates that an increase in the number of cars (logCAR) will lead to an increase in pollution levels. Conversely, a decrease in the number of cars will lead to a decrease in pollution levels.

Negative coefficients: When the coefficient of an input attribute is negative, it means that an increase in the value of that attribute will result in a decrease in the output value. In this case, the negative coefficient for x3 (WIND) indicates that an increase in wind speed (WIND) will lead to a decrease in pollution levels. Conversely, a decrease in wind speed will lead to an increase in pollution levels.

So, the conclusion for statement B is based on understanding that the positive coefficient for x1 (logCAR) indicates that fewer cars will result in lower pollution levels, and the negative coefficient for x3 (WIND) indicates that more wind will result in lower pollution levels.

C. According to the model, it seems that pollution is decreasing over time.

This statement is incorrect. The model has a positive coefficient for x7 (DAY), which means that pollution is increasing over time, not decreasing.

D. According to the model, higher temperatures will result in lower pollution levels.

This statement is correct. The model has a negative coefficient for x2 (TEMP), meaning that higher temperatures are associated with lower pollution levels.

Therefore, the incorrect statement is C.
"""
import numpy as np


def linear_regression_model(x):
    x1, x2, x3, x4, x7 = x
    return 3.27 + 0.36 * x1 - 0.01 * x2 - 0.19 * x3 + 0.01 * x4 + 0.05 * x7


def statement_A():
    return "WINDDIR" not in linear_regression_model.__code__.co_varnames and "HOUR" not in linear_regression_model.__code__.co_varnames


def statement_B(x1_diff, x3_diff):
    initial_pollution = linear_regression_model((0, 0, 0, 0, 0))
    new_pollution = linear_regression_model((-x1_diff, 0, x3_diff, 0, 0))
    return new_pollution < initial_pollution


def statement_C(x7_diff):
    initial_pollution = linear_regression_model((0, 0, 0, 0, 0))
    new_pollution = linear_regression_model((0, 0, 0, 0, x7_diff))
    return new_pollution < initial_pollution


def statement_D(x2_diff):
    initial_pollution = linear_regression_model((0, 0, 0, 0, 0))
    new_pollution = linear_regression_model((0, x2_diff, 0, 0, 0))
    return new_pollution < initial_pollution


print('8.1')
# Test each statement
print("A:", statement_A())
print("B:", statement_B(1, 1))
print("C:", statement_C(1))
print("D:", statement_D(1))
print('\n')

# 2
"""
Let's revisit the statements based on the given logistic regression model:

f(x1, x2, x3, x4, x5, x6) = logit(-0.51 - 0.11x1 - 0.36x2 + 0.44x3 + 0.39x4 + 0.09x5 - 0.28x6)

A. According to the model, it seems that people in Lisbon buy more FRESH products, MILK products, and DELI products than people in Oporto.

This statement is correct. Since x1 (FRESH), x2 (MILK), and x6 (DELI) have negative coefficients, larger values for these attributes will result in a lower probability of a consumer being from Oporto (y=1). Therefore, people in Lisbon are more likely to have higher consumption of FRESH, MILK, and DELI products than people in Oporto.

B. According to the model, if a customer after standardization has x1 = x2 = x3 = x4 = x5 = x6 = 0, the customer is more likely to come from Oporto than Lisbon.

This statement is incorrect. If all standardized input attributes are 0, the model simplifies to:

logit(-0.51) = 1 / (1 + exp(0.51)) ≈ 0.375

Since the logistic function returns the probability of a customer being from Oporto, there is a 37.5% chance that the customer is from Oporto, and a 62.5% chance that the customer is from Lisbon. Therefore, the customer is more likely to come from Lisbon than Oporto.

C. The logit function will return the probability a person is from Lisbon.

This statement is incorrect. The logit function in this model returns the probability that a person is from Oporto, not Lisbon. To calculate the probability that a person is from Lisbon, you would use the complementary probability: 1 - logit(f(x1, x2, x3, x4, x5, x6)).

D. From the model, it can be seen that FRESH and PAPER are unimportant and should be removed in order to avoid overfitting.

This statement is incorrect. While the coefficients of FRESH and PAPER are the smallest, they may still contribute to the predictions. To determine whether these attributes should be removed from the modeling, the test performance including FRESH and PAPER should be compared to not including these two attributes.

The correct statement is A.

"""

import numpy as np


def logit(w):
    return 1 / (1 + np.exp(-w))


def logistic_regression_model(x1, x2, x3, x4, x5, x6):
    w = -0.51 - 0.11 * x1 - 0.36 * x2 + 0.44 * x3 + 0.39 * x4 + 0.09 * x5 - 0.28 * x6
    return logit(w)


# Example standardized input values
x1 = 0
x2 = 0
x3 = 0
x4 = 0
x5 = 0
x6 = 0

# Calculate the probability of a customer being from Oporto
probability_oporto = logistic_regression_model(x1, x2, x3, x4, x5, x6)

# Calculate the probability of a customer being from Lisbon
probability_lisbon = 1 - probability_oporto

print('8.2')
print(f"Probability of customer being from Oporto: {probability_oporto:.3f}")
print(f"Probability of customer being from Lisbon: {probability_lisbon:.3f}")
print('\n')

# 3
"""
Let's reanalyze the given statements:

A. According to the model, AreaNI is irrelevant for predicting the Area of islands.

This statement is incorrect. As mentioned before, the model has a non-zero coefficient for x7 (AreaNI), which means it is considered relevant for predicting the area of islands.

B. According to the model, it seems that the closer the neighboring island is, the larger area the island has.

This statement is correct. As previously mentioned, the model has a negative coefficient for x5 (DistNI), which means that an increase in x5 (Distance to the nearest island) is associated with a decrease in the predicted area of the island (x3). Thus, the model suggests that the closer the neighboring island is, the larger the area the island has.

C. According to the model, endemic plants are the most important predictor of island area.

This statement is incorrect. Although the coefficient for x2 (E-Plants) is the largest in magnitude, the range of this attribute is limited, as seen from Figure 8.10. It is thus not reasonable to say this is the most important attribute for predicting the area.

D. According to the model, an island that is highly elevated and close to Santa Cruz Island will, in general, be predicted to be relatively small.

This statement is incorrect. As previously mentioned, the model has a positive coefficient for x4 (Elev) and a negative coefficient for x6 (StCruz). An island with a high elevation (larger x4) will be predicted to have a larger area, while an island close to Santa Cruz Island (smaller x6) will also be predicted to have a larger area.

The correct statement is B.
"""


def linear_regression_model(x1, x2, x4, x5, x6, x7):
    return 63.4 + 4.3 * x1 - 34.7 * x2 + 3.0 * x4 - 7.2 * x5 - 1.4 * x6 - 0.5 * x7


# Example input values (replace with actual values as needed)
x1 = 10
x2 = 5
x4 = 1000
x5 = 20
x6 = 30
x7 = 40

# Calculate the predicted area of the island
predicted_area = linear_regression_model(x1, x2, x4, x5, x6, x7)

print('8.3')
print(f"Predicted area of the island: {predicted_area:.2f} km^2")
print('\n')

# 4
"""
To analyze the given confusion matrices, we first need to calculate the precision, recall, true negative rate, and false positive rate for both classifiers.

Confusion Matrix format:
TP (True Positive) | FP (False Positive)
FN (False Negative) | TN (True Negative)

Logistic Regression Confusion Matrix:
12 | 69
10 | 215

Decision Tree Confusion Matrix:
26 | 55
34 | 191

Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
True Negative Rate (TNR) = TN / (TN + FP)
False Positive Rate (FPR) = FP / (FP + TN)

Logistic Regression Metrics:
Precision = 12 / (12 + 69) = 0.148
Recall = 12 / (12 + 10) = 0.545
TNR = 215 / (215 + 69) = 0.757
FPR = 69 / (69 + 215) = 0.243

Decision Tree Metrics:
Precision = 26 / (26 + 55) = 0.321
Recall = 26 / (26 + 34) = 0.433
TNR = 191 / (191 + 55) = 0.776
FPR = 55 / (55 + 191) = 0.224

A. The precision of the logistic regression classifier is higher than the precision of the decision tree classifier.
Correct. The precision of the logistic regression classifier is 0.545, and the decision tree classifier is 0.433.

B. The recall of the logistic regression classifier is higher than the recall of the decision tree classifier.
Incorrect. The recall of the logistic regression classifier is 0.148, and the decision tree classifier is 0.321.

C. The true negative rate of the logistic regression classifier is lower than the true negative rate of the decision tree classifier.
Incorrect. The true negative rate of the logistic regression classifier is 0.956, and the decision tree classifier is 0.849.

D. The false positive rate of the logistic regression classifier is higher than the false positive rate of the decision tree classifier.
Incorrect. The false positive rate of the logistic regression classifier is 0.044, and the decision tree classifier is 0.151.

The correct statement is A.

"""


def classifier_metrics(tp, fp, fn, tn):
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    tnr = tn / (tn + fp)
    fpr = fp / (tn + fp)
    return precision, recall, tnr, fpr


# Logistic Regression
logistic_regression_confusion_matrix = {
    'tp': 12, 'fp': 10, 'fn': 69, 'tn': 215
}

# Decision Tree
decision_tree_confusion_matrix = {
    'tp': 26, 'fp': 34, 'fn': 55, 'tn': 191
}

logistic_regression_metrics = classifier_metrics(**logistic_regression_confusion_matrix)
decision_tree_metrics = classifier_metrics(**decision_tree_confusion_matrix)

print('8.4')
print("Logistic Regression Metrics:")
print(f"Precision: {logistic_regression_metrics[0]:.3f}")
print(f"Recall: {logistic_regression_metrics[1]:.3f}")
print(f"TNR: {logistic_regression_metrics[2]:.3f}")
print(f"FPR: {logistic_regression_metrics[3]:.3f}")

print("\nDecision Tree Metrics:")
print(f"Precision: {decision_tree_metrics[0]:.3f}")
print(f"Recall: {decision_tree_metrics[1]:.3f}")
print(f"TNR: {decision_tree_metrics[2]:.3f}")
print(f"FPR: {decision_tree_metrics[3]:.3f}")
print('\n')

# 5
"""
The goal is to find the correct values for w0, w1, w2, and w3 for the given logistic regression classifier. We have four options to choose from (A, B, C, and D).

1. We first look at the logistic regression function value at the point (x1, x2) = (0, 0) for options A, B, C, and D. 
   We find that options A and D produce a value of 0.88, while options B and C produce a value of 0.12. Based on the bottom pane of Fig. 8.12, 
   we can see that the estimated probability of belonging to class 1 at the point (0, 0) is closer to 0.88. So, we can eliminate options B and C.
    
We can see from the bottom pane of Fig. 8.12 that at the point (0, 0), the estimated probability of belonging to class 1 is high (closer to 1). When we calculated the logistic function value at (0, 0) for options A, B, C, and D, we found that options A and D produce a value of 0.88, which is a high probability. On the other hand, options B and C produce a value of 0.12, which is a low probability.

Since the figure shows a high probability at point (0, 0), we can conclude that options A and D are more likely to be the correct answer as they produce high probabilities at that point. Therefore, we can eliminate options B and C, as they produce low probabilities at (0, 0) which do not match the probability distribution observed in the figure.


2. Now we have two options left: A and D. To differentiate between them, we analyze the density estimate at the point (1, 1) for option D. 
   The density estimate is approximately 0.0025, which is too low for the high-density corner observed in the bottom pane of Fig. 8.12.

Now we have two options left: A and D. Both of them produce a high probability at the point (0, 0), as we saw earlier. To differentiate between them, we need to analyze their behavior at another point, such as (1, 1). We choose this point because it is in the "high-density" corner of the bottom pane of Fig. 8.12, meaning it should have a high probability of belonging to class 1.

Let's calculate the estimated probability of belonging to class 1 at the point (1, 1) for option D:

f(x1, x2) = logistic(w0 + w1x1 + w2x2 + w3x1x2) = logistic(2 + 11 + 11 - 1011)

f(1, 1) = logistic(2 + 1 + 1 - 10) = logistic(-6)

Using the logistic function, the estimated probability is:

1 / (1 + e^6) ≈ 0.0025

For option D, the estimated probability of belonging to class 1 at the point (1, 1) is around 0.0025, which is a very low probability. However, when we look at the bottom pane of Fig. 8.12, we can see that the probability at point (1, 1) should be high. This means that option D is not consistent with the observed probability distribution, and we can rule it out.

This leaves us with option A as the correct answer since it is the only remaining option and it produces a high probability at both points (0, 0) and (1, 1), which is consistent with the probability distribution observed in the bottom pane of Fig. 8.12.

Conclusion: 
So, after analyzing the given data, we conclude that the correct answer is option A, with w0 = 2, w1 = w2 = 0, and w3 = 10.

"""
import numpy as np

def logistic(z):
    return 1 / (1 + np.exp(-z))

# The point we want to evaluate
point = (1, 1)

# Option A parameters
w0_A, w1_A, w2_A, w3_A = 2, 0, 0, 10

# Option D parameters
w0_D, w1_D, w2_D, w3_D = 2, 1, 1, -10

# Calculate the logistic function output for the point (1, 1) using option A parameters
prob_A = logistic(w0_A + w1_A * point[0] + w2_A * point[1] + w3_A * point[0] * point[1])

# Calculate the logistic function output for the point (1, 1) using option D parameters
prob_D = logistic(w0_D + w1_D * point[0] + w2_D * point[1] + w3_D * point[0] * point[1])

print('8.5')
print("Estimated probability of belonging to class 1 at point (1, 1) for option A:", prob_A)
print("Estimated probability of belonging to class 1 at point (1, 1) for option D:", prob_D)


# The point we want to evaluate
point = (0, 0)

# Parameters for all the options
params = {
    "A": (2, 0, 0, 10),
    "B": (-2, 0, 0, -10),
    "C": (-2, 1, 1, 10),
    "D": (2, 1, 1, -10)
}

# Calculate and print the logistic function output for the point (0, 0) using all options' parameters
for option, (w0, w1, w2, w3) in params.items():
    prob = logistic(w0 + w1 * point[0] + w2 * point[1] + w3 * point[0] * point[1])
    print(f"Estimated probability of belonging to class 1 at point (0, 0) for option {option}: {prob}")
print('\n')

######################################
# 09
######################################
print("Section 09:")
# 1
"""
To calculate the gain of splitting according to whether a subject had positive axillary nodes (PAN), we first need to calculate the Gini impurity for the parent node and the weighted Gini impurity for the child nodes. Then, we can find the gain by subtracting the weighted Gini impurity of the child nodes from the Gini impurity of the parent node.

Calculate Gini impurity for the parent node:
I(parent) = 1 - [(81/306)^2 + (225/306)^2]

Calculate Gini impurity for the child nodes:

For subjects with PAN (170 subjects, 62 survived):
I(PAN = 1) = 1 - [(62/170)^2 + (108/170)^2]
For subjects without PAN (136 subjects, 19 survived):
I(PAN = 0) = 1 - [(19/136)^2 + (117/136)^2]
Calculate the weighted Gini impurity for the child nodes:
Weighted Gini = (170/306) * I(PAN = 1) + (136/306) * I(PAN = 0)

Calculate the gain:
Gain = I(parent) - Weighted Gini

Now let's calculate these values in Python:
"""
parent_survived = 81
parent_died = 225
parent_total = parent_survived + parent_died

PAN1_survived = 62
PAN1_total = 170
PAN0_survived = 19
PAN0_total = 136

I_parent = 1 - ((parent_survived / parent_total) ** 2 + (parent_died / parent_total) ** 2)

I_PAN1 = 1 - ((PAN1_survived / PAN1_total) ** 2 + ((PAN1_total - PAN1_survived) / PAN1_total) ** 2)
I_PAN0 = 1 - ((PAN0_survived / PAN0_total) ** 2 + ((PAN0_total - PAN0_survived) / PAN0_total) ** 2)

weighted_gini_child = (PAN1_total / parent_total) * I_PAN1 + (PAN0_total / parent_total) * I_PAN0

gain = I_parent - weighted_gini_child

print('9.1')
print(gain)
print('\n')

# 2
"""
The problem is asking which of the given decision boundaries (A, B, C, D) can best separate the two classes (red crosses and black circles) in the given classification problem (Fig. 9.8).

Answer A consists of two decision boundaries:

Decision A: ||[x1, x2] - [0.5, 0.25]||∞ < 0.25
This decision uses the infinity norm, which creates a square-shaped decision boundary around the point (0.5, 0.25). This boundary captures the red crosses close to the point (0.5, 0.25).

Decision B: ||[x1, x2] - [0.5, 0.5]||2 < 0.25
This decision uses the L2 (Euclidean) norm, which creates a circular decision boundary around the point (0.5, 0.5). This boundary separates the circular shape of red crosses in the upper middle from all the remaining observations that are black circles.

These two decision boundaries together (Answer A) form the best separation between the two classes (red crosses and black circles) as compared to the other answer choices (B, C, D).

In the expression A = ||[x1, x2] - [0.5, 0.25]||1 < 0.25, the number "1" represents the L1 norm (also known as the Manhattan distance). The L1 norm is the sum of the absolute differences between two points in a coordinate space. In this case, the L1 norm is calculated as:

L1 norm = |x1 - 0.5| + |x2 - 0.25|

The expression states that the sum of the absolute differences between the point [x1, x2] and the reference point [0.5, 0.25] should be less than 0.25 for the decision boundary. When the L1 norm is used, the decision boundary forms a diamond shape around the reference point.

---
L1, L2, and infinity norms are different ways of measuring distances between points in a coordinate space. Here's a brief explanation of each, along with examples:

L1 norm (Manhattan distance):
The L1 norm is the sum of the absolute differences between the coordinates of two points. It is called the Manhattan distance because it resembles the distance you would travel in a city with a grid-like street layout, like Manhattan.

Example:
Point A: (1, 2)
Point B: (4, 6)

L1(A, B) = |1 - 4| + |2 - 6| = 3 + 4 = 7

L2 norm (Euclidean distance):
The L2 norm is the square root of the sum of the squared differences between the coordinates of two points. It is the most common distance metric and corresponds to the straight-line distance between two points.

Example:
Point A: (1, 2)
Point B: (4, 6)

L2(A, B) = sqrt((1 - 4)^2 + (2 - 6)^2) = sqrt(9 + 16) = sqrt(25) = 5

Infinity norm (Chebyshev distance):
The infinity norm is the maximum absolute difference between the coordinates of two points. It is called the Chebyshev distance because it corresponds to the number of moves a king would make in a game of chess to travel between two squares.

Example:
Point A: (1, 2)
Point B: (4, 6)

L-infinity(A, B) = max(|1 - 4|, |2 - 6|) = max(3, 4) = 4

In summary, L1, L2, and infinity norms are different ways of measuring distances between points, with each norm having unique properties and geometric interpretations.
---
Here are the geometric interpretations of L1, L2, and infinity norms:

L1 norm (Manhattan distance):
The L1 norm corresponds to the distance traveled along the grid lines in a coordinate space. In a 2D plane, the L1 distance between two points forms a diamond-shaped pattern. In this case, the distance is the sum of the absolute differences between the coordinates of two points.

L2 norm (Euclidean distance):
The L2 norm corresponds to the straight-line distance between two points in a coordinate space. In a 2D plane, the L2 distance forms a circular pattern. The L2 norm is the most common distance metric, and it is derived from the Pythagorean theorem.

Infinity norm (Chebyshev distance):
The infinity norm corresponds to the maximum absolute difference between the coordinates of two points. In a 2D plane, the infinity distance forms a square-shaped pattern. This distance is related to the number of moves a king would make in a game of chess to travel between two squares.

There are other norms, such as the general Lp norm, which can be used to derive various distance metrics by changing the value of p:

Lp norm: Lp(A, B) = (|x1 - y1|^p + |x2 - y2|^p + ... + |xn - yn|^p)^(1/p)

For example:

L0.5 norm: This is a fractional norm, which results in a more "pointy" shape compared to the L1 and L2 norms.
L3 norm: This norm generates a more "rounded" shape compared to the L2 norm but is less commonly used.
In summary, different norms have different geometric interpretations, and the choice of norm can have a significant impact on the measurement of distances and the behavior of algorithms that rely on distance measurements.
"""
# Just an example
print('9.2')
import numpy as np

# Define two points A and B
A = np.array([3, 4])
B = np.array([6, 8])

# Calculate L1 distance (Manhattan distance)
L1_distance = np.sum(np.abs(A - B))
print(f"L1 distance: {L1_distance}")

# Calculate L2 distance (Euclidean distance)
L2_distance = np.sqrt(np.sum(np.square(A - B)))
print(f"L2 distance: {L2_distance}")

# Calculate Infinity distance (Chebyshev distance)
infinity_distance = np.max(np.abs(A - B))
print(f"Infinity distance: {infinity_distance}")

# Calculate Lp distance (generalized distance)
p = 0.5
Lp_distance = np.power(np.sum(np.power(np.abs(A - B), p)), 1 / p)
print(f"L{p} distance: {Lp_distance}")

p = 3
Lp_distance = np.power(np.sum(np.power(np.abs(A - B), p)), 1 / p)
print(f"L{p} distance: {Lp_distance}")
print('\n')

# 3
"""
We need to compute the impurity for the entire dataset, the lower part, and the upper part using classification error as the impurity measure.

Before the split:

Impurity_before_split = 1 - max(3/7, 1/7, 3/7) = 1 - 3/7 = 4/7
After the split (X = x1 > 2.5):

Lower part (X <= 2.5): 5 observations

Impurity_left = 1 - max(3/5, 0, 2/5) = 1 - 3/5 = 2/5
Upper part (X > 2.5): 2 observations

Impurity_right = 1 - max(1/2, 0, 1/2) = 1 - 1/2 = 1/2
Weighted average impurity after split:

Impurity_after_split = (5/7) * Impurity_left + (2/7) * Impurity_right = (5/7) * (2/5) + (2/7) * (1/2) = 2/7 + 1/7 = 3/7
Impurity gain:

∆ = Impurity_before_split - Impurity_after_split = 4/7 - 3/7 = 1/7
The impurity gain is roughly 1/7, which is approximately 0.143. So, the correct answer is B.
"""
import numpy as np

# Dataset
X = np.array([1, 3, 1, 2, 1, 4, 2])
y = np.array([2, 2, 2, 0, 0, 1, 0])

# Calculate class probabilities
classes, counts = np.unique(y, return_counts=True)
total_samples = len(y)
class_probs = counts / total_samples

# Calculate impurity before the split
impurity_before_split = 1 - np.max(class_probs)

# Split the dataset
split_value = 2.5
mask = X > split_value
left_y = y[~mask]
right_y = y[mask]

# Calculate impurity for the left part
left_classes, left_counts = np.unique(left_y, return_counts=True)
left_samples = len(left_y)
left_class_probs = left_counts / left_samples
impurity_left = 1 - np.max(left_class_probs)

# Calculate impurity for the right part
right_classes, right_counts = np.unique(right_y, return_counts=True)
right_samples = len(right_y)
right_class_probs = right_counts / right_samples
impurity_right = 1 - np.max(right_class_probs)

# Calculate the weighted average impurity after the split
impurity_after_split = (left_samples / total_samples) * impurity_left + (right_samples / total_samples) * impurity_right

# Calculate impurity gain
impurity_gain = impurity_before_split - impurity_after_split

print('9.3')
print(f"Impurity gain: {impurity_gain:.3f}")
print('\n')

# 4
"""
                 A: ||x||2 >= 1/2
            ┌────────────┴────────────┐
       true                           false
            B: ||x - [-1, 1]||∞ > 1     C: ||x - [1, -1]||1 > 2
     ┌─────────────┴─────────────┐   ┌─────────────┴─────────────┐
  true                       false   true                       false
Class 1                      Class 0 Class 1                    Class 0

The goal is to find a decision tree that can accurately classify the data points based on their coordinates (x1, x2).

The correct decision tree structure is shown in Figure 9.11, and the answer A provides the correct rules for nodes A, B, and C in the decision tree. These rules create the decision boundaries that can accurately classify the data points.

Here are the rules for nodes A, B, and C in the decision tree:

A: ||x||2 >= 1/2
B: ||x - [-1, 1]||∞ > 1
C: ||x - [1, -1]||1 > 2

Now, let's go through the rules one by one:

Rule A: ||x||2 >= 1/2
This rule checks whether the Euclidean distance (L2-norm) of the data point 'x' from the origin is greater than or equal to 1/2. If true, the point belongs to Class 1.

Rule B: ||x - [-1, 1]||∞ > 1
This rule checks whether the Chebyshev distance (L-infinity norm) of the data point 'x' from the point (-1, 1) is greater than 1. If true, the point belongs to Class 0.

Rule C: ||x - [1, -1]||1 > 2
This rule checks whether the Manhattan distance (L1-norm) of the data point 'x' from the point (1, -1) is greater than 2. If true, the point belongs to Class 0.
----
When evaluating the decision trees at the corners (-1, 1) and (1, -1), only options A and C give correct classifications for these points. However, if we further examine the decision boundary and consider the point (-0.2, 0.2) that lies inside the circle, option A correctly classifies it as class 0, while option C does not. This discrepancy shows that option C is not the correct choice, and option A is the correct answer.
"""

import numpy as np


def decision_tree_A(x):
    A = np.linalg.norm(x, 2) >= 1 / 2
    B = np.linalg.norm(x - np.array([-1, 1]), np.inf) > 1
    C = np.linalg.norm(x - np.array([1, -1]), 1) > 2

    if A:
        if B:
            return 1  # Class 1
        else:
            return 0  # Class 0
    else:
        if C:
            return 1  # Class 1
        else:
            return 0  # Class 0


def decision_tree_B(x):
    A = np.linalg.norm(x, 2) >= 1 / 2
    B = np.linalg.norm(x - np.array([1, -1]), 1) > 2
    C = np.linalg.norm(x - np.array([-1, 1]), np.inf) > 1

    if A:
        if B:
            return 1  # Class 1
        else:
            return 0  # Class 0
    else:
        if C:
            return 1  # Class 1
        else:
            return 0  # Class 0


def decision_tree_C(x):
    A = np.linalg.norm(x - np.array([1, -1]), 1) > 2
    B = np.linalg.norm(x - np.array([-1, 1]), np.inf) > 1
    C = np.linalg.norm(x, 2) >= 1 / 2

    if A:
        if B:
            return 1  # Class 1
        else:
            return 0  # Class 0
    else:
        if C:
            return 1  # Class 1
        else:
            return 0  # Class 0


def decision_tree_D(x):
    A = np.linalg.norm(x - np.array([1, -1]), 1) > 2
    B = np.linalg.norm(x, 2) >= 1 / 2
    C = np.linalg.norm(x - np.array([-1, 1]), np.inf) > 1

    if A:
        if B:
            return 1  # Class 1
        else:
            return 0  # Class 0
    else:
        if C:
            return 1  # Class 1
        else:
            return 0  # Class 0


# Test the decision tree with some sample points
sample_points = [
    (-1, 1),
    (1, -1),
    (-0.5, 0.5),
    (0.5, -0.5),
    (0, 0),
    (0, 0.5),
    (-0.5, 0),
    (0.5, 0),
    (0, -0.5),
    (-0.2, 0.2)
]

print('9.4')
print("Decision tree A:")
for point in sample_points:
    x = np.array(point)
    print(f"Point: {point}, Class: {decision_tree_A(x)}")

print("\nDecision tree B:")
for point in sample_points:
    x = np.array(point)
    print(f"Point: {point}, Class: {decision_tree_B(x)}")

print("\nDecision tree C:")
for point in sample_points:
    x = np.array(point)
    print(f"Point: {point}, Class: {decision_tree_C(x)}")

print("\nDecision tree D:")
for point in sample_points:
    x = np.array(point)
    print(f"Point: {point}, Class: {decision_tree_D(x)}")
print('\n')

######################################
# 10
######################################
print("Section 10:")
# 1
"""
The total time taken for the entire procedure according to the given information.

1- Cross-validation on DCV: Alice uses 10-fold cross-validation on 80% of the dataset (since Dvalidation is 20%).

    - N_CV = 0.8 * N = 0.8 * 1000 = 800
    - Each fold consists of 800 / 10 = 80 observations.
    - For each of the 6 regularization strengths, Alice trains on 9 folds (9 * 80 = 720 observations) and tests on 1 fold (80 observations).
    - Training time per regularization strength: 720^2
    - Testing time per regularization strength: 0.5 * 80^2
    - Total time for cross-validation: 6 * 10 * (training time + testing time) = 6 * 10 * (720^2 + 0.5 * 80^2)

2- Hold-out method:

    - Alice trains the model with the optimal regularization strength on DCV (80% of the dataset) and tests on Dvalidation (20% of the dataset).
    - Training time: 800^2
    - Testing time: 0.5 * 200^2
    
Now, let's calculate the total time taken for the entire procedure:
"""
cross_val_training_time = 720 ** 2
cross_val_testing_time = 0.5 * 80 ** 2
hold_out_training_time = 800 ** 2
hold_out_testing_time = 0.5 * 200 ** 2

total_cross_val_time = 6 * 10 * (cross_val_training_time + cross_val_testing_time)
total_hold_out_time = hold_out_training_time + hold_out_testing_time
total_time = total_cross_val_time + total_hold_out_time

print('10.1')
print("Total time taken for the entire procedure:", total_time)
print('\n')

# 2
"""
To determine the correct statement, let's analyze the root mean square error (RMSE) values in Table 10.4.

Forward selection starts with an empty set of features and iteratively adds the feature that results in the lowest RMSE for the test set.

First, we select the feature with the lowest test RMSE among the individual features (x1, x2, x3, and x4). In this case, it's x3 with a test RMSE of 0.62.
Then, we check the combinations of x3 with the other remaining features (x1, x2, and x4). The combination with the lowest test RMSE is x2 and x3 with a test RMSE of 0.69.
Finally, we check the combinations of x2 and x3 with the remaining features (x1 and x4). The combination with the lowest test RMSE is x1, x2, and x3 with a test RMSE of 0.34.
So, forward selection selects the features x1, x2, and x3.

Backward selection starts with the full set of features and iteratively removes the feature that results in the lowest RMSE for the test set.

First, we remove one feature at a time from the full set of features (x1, x2, x3, and x4) and select the combination with the lowest test RMSE. In this case, it's x1, x2, and x3 with a test RMSE of 0.34.
Then, we remove one feature at a time from the combination x1, x2, and x3. The combination with the lowest test RMSE is still x1, x2, and x3 with a test RMSE of 0.34.
So, backward selection selects the features x1, x2, and x3.

Based on this analysis, the correct statement is:

B. Backward selection will result in a better model being selected than using forward selection.
"""
import pandas as pd
import numpy as np
from itertools import combinations

# Data from Table 10.4
data = {
    'features': [
        'x1', 'x2', 'x3', 'x4',
        'x1,x2', 'x1,x3', 'x1,x4',
        'x2,x3', 'x2,x4', 'x3,x4',
        'x1,x2,x3', 'x1,x2,x4', 'x1,x3,x4', 'x2,x3,x4',
        'x1,x2,x3,x4'
    ],
    'training_rmse': [0.71, 0.58, 0.60, 0.92, 0.60, 0.35, 0.52, 0.56, 0.45, 0.62, 0.36, 0.28, 0.27, 0.20, 0.10],
    'test_rmse': [0.75, 0.64, 0.62, 0.94, 0.69, 0.44, 0.66, 0.69, 0.52, 0.64, 0.34, 0.33, 0.45, 0.43, 0.35],
}

df = pd.DataFrame(data)


def forward_selection(df):
    features = ['x1', 'x2', 'x3', 'x4']
    selected_features = []
    min_rmse = float('inf')

    for _ in range(len(features)):
        best_feature = None

        for feature in features:
            if feature in selected_features:
                continue

            current_features = selected_features + [feature]
            feature_set = ','.join(sorted(current_features))
            rmse = df.loc[df['features'] == feature_set, 'test_rmse'].values[0]

            if rmse < min_rmse:
                min_rmse = rmse
                best_feature = feature

        if best_feature is not None:
            selected_features.append(best_feature)

    return selected_features


def backward_selection(df):
    features = ['x1', 'x2', 'x3', 'x4']
    selected_features = features.copy()
    min_rmse = df.loc[df['features'] == ','.join(sorted(selected_features)), 'test_rmse'].values[0]

    while len(selected_features) > 1:
        best_rmse = float('inf')
        feature_to_remove = None

        for feature in selected_features:
            current_features = [f for f in selected_features if f != feature]
            feature_set = ','.join(sorted(current_features))
            rmse = df.loc[df['features'] == feature_set, 'test_rmse'].values[0]

            if rmse < best_rmse:
                best_rmse = rmse
                feature_to_remove = feature

        if best_rmse < min_rmse:
            min_rmse = best_rmse
            selected_features.remove(feature_to_remove)
        else:
            break

    return selected_features


forward_selected_features = forward_selection(df)
backward_selected_features = backward_selection(df)

print('10.2')
print('Forward selected features:', forward_selected_features)
print('Backward selected features:', backward_selected_features)
print('\n')

# 3
"""
A - Correct. Cross-validation can be used to quantify a model's generalization error by dividing the dataset into 
multiple folds, training the model on a subset of the data, and testing it on the remaining data. This process is 
repeated for all folds and the average error is used as an estimate of the model's generalization error. 

B - Correct. K-fold cross-validation divides the dataset into K equal-sized folds. When K = N, where N is the total 
number of observations, K-fold cross-validation becomes leave-one-out cross-validation, where each observation is 
used as a test set exactly once, and the remaining observations are used for training. 

C - Correct. When using cross-validation to estimate model parameters, such as hyperparameters, an extra level of 
cross-validation is needed to evaluate the model's generalization performance. This is because the model performance 
estimated during hyperparameter tuning may be overly optimistic, as the model is tuned specifically to perform well 
on the validation set. By using an additional level of cross-validation, we can get a more accurate estimate of the 
model's generalization error. 

D - Incorrect. For least squares linear regression, the test error will not always decrease as we include more 
attributes in the model. Including more attributes can lead to overfitting, where the model becomes too complex and 
fits the noise in the training data. This results in a decrease in the model's ability to generalize to new, 
unseen data. While the training error may decrease as more attributes are included, the test error may increase due 
to overfitting. 
"""

# 4
"""
A - Incorrect. Forward or backward selection can be used to identify the optimal feature combination for this problem. By analyzing the training and test RMSE in Table 10.6, we can determine which feature combinations perform better on the test set.

B - Incorrect. Based on the analysis, forward selection will result in the features x2 and x5, while backward selection will terminate with features x2 and x6. Although the model with features x2 and x6 has a lower test RMSE (22.5) than the combination of features x2 and x5 (36.3), we cannot say that one method is better than the other without considering other factors such as model complexity and interpretability.

C - Incorrect. As mentioned earlier, backward selection will terminate at the model that includes features x2 and x6, not x1, x2, and x6.

D - Correct. Forward selection will select the features x2 and x5. First, it will select x5, as it has the lowest test RMSE as a single feature. Then, it will select x2, as it decreases the test RMSE the most when combined with x5. Further selecting features in combination with x2 and x5 does not improve the test error, so the model will terminate.
"""
import pandas as pd
import numpy as np
from itertools import combinations

# Data from Table 10.6
data = {
    'features': [
        'x1', 'x2', 'x5', 'x6',
        'x1,x2', 'x1,x5', 'x1,x6',
        'x2,x5', 'x2,x6', 'x5,x6',
        'x1,x2,x5', 'x1,x2,x6', 'x1,x5,x6', 'x2,x5,x6',
        'x1,x2,x5,x6'
    ],
    'training_rmse': [81.2, 62.3, 68.0, 98.9, 57.1, 40.2, 55.2, 32.2, 20.3, 48.4, 36.6, 18.8, 33.3, 40.4, 30.0],
    'test_rmse': [80.1, 84.3, 72.9, 100.5, 69.1, 43.3, 66.4, 36.3, 22.5, 50.3, 39.1, 23.0, 36.7, 43.0, 35.2],
}

df = pd.DataFrame(data)


def forward_selection(df):
    features = ['x1', 'x2', 'x5', 'x6']
    selected_features = []
    min_rmse = float('inf')

    for _ in range(len(features)):
        best_feature = None

        for feature in features:
            if feature in selected_features:
                continue

            current_features = selected_features + [feature]
            feature_set = ','.join(sorted(current_features))
            rmse = df.loc[df['features'] == feature_set, 'test_rmse'].values[0]

            if rmse < min_rmse:
                min_rmse = rmse
                best_feature = feature

        if best_feature is not None:
            selected_features.append(best_feature)

    return selected_features


def backward_selection(df):
    features = ['x1', 'x2', 'x5', 'x6']
    selected_features = features.copy()
    min_rmse = df.loc[df['features'] == ','.join(sorted(selected_features)), 'test_rmse'].values[0]

    while len(selected_features) > 1:
        best_rmse = float('inf')
        feature_to_remove = None

        for feature in selected_features:
            current_features = [f for f in selected_features if f != feature]
            feature_set = ','.join(sorted(current_features))
            rmse = df.loc[df['features'] == feature_set, 'test_rmse'].values[0]

            if rmse < best_rmse:
                best_rmse = rmse
                feature_to_remove = feature

        if best_rmse < min_rmse:
            min_rmse = best_rmse
            selected_features.remove(feature_to_remove)
        else:
            break

    return selected_features


forward_selected_features = forward_selection(df)
backward_selected_features = backward_selection(df)

print('10.4')
print('Forward selected features:', forward_selected_features)
print('Backward selected features:', backward_selected_features)
print('\n')

# 5
"""
A. The error rate of the logistic regression classifier is larger than the error rate of the decision tree classifier.

Error rate is calculated as the ratio of misclassified observations to the total number of observations.

For Logistic Regression:
Error rate = (12 + 10) / 306 = 22 / 306 = 0.0719 (approximately)

For Decision Tree:
Error rate = (26 + 34) / 306 = 60 / 306 = 0.1961 (approximately)

Since 0.0719 < 0.1961, the error rate of the logistic regression classifier is indeed smaller, not larger, than the error rate of the decision tree classifier. So, statement A is incorrect.

B. Logistic regression:

Error rate: (10 + 69) / 306 = 79 / 306 ≈ 0.258
Accuracy: (215 + 12) / 306 = 227 / 306 ≈ 0.742
Decision tree:

Error rate: (34 + 55) / 306 = 89 / 306 ≈ 0.291
Accuracy: (191 + 26) / 306 = 217 / 306 ≈ 0.709
In conclusion, the logistic regression classifier has a smaller error rate and higher accuracy than the decision tree classifier. Due to the imbalance in the classes, predicting all cases as belonging to the "died" class would yield an accuracy of 225 / 306 ≈ 0.735, which is higher than the accuracy obtained by the decision tree classifier. Therefore, the correct answer is B: Predicting every observation to be in the died class would give a better accuracy than the accuracy obtained by the decision tree classifier.

C. The classification problem does not have any issues of imbalanced classes.

Imbalanced classes occur when one class has significantly more observations than the other class. In this case, there are 81 survived observations and 225 died observations, making the dataset imbalanced. So, statement C is incorrect.

D. Using leave-one-out cross validation, a total of 306 − 1 = 305 logistic regression models are trained.

In leave-one-out cross-validation (LOOCV), the dataset is split into N folds, where N is the total number of observations in the dataset. One observation is used as the test set, and the remaining N-1 observations are used as the training set. This process is repeated for all N observations in the dataset.

In this case, N = 306. So, a total of 306 logistic regression models will be trained, not 305. Statement D is incorrect.
"""

# Confusion matrices
logistic_regression_cm = [[12, 69], [10, 215]]
decision_tree_cm = [[26, 55], [34, 191]]


# Calculate error rates and accuracies
def calculate_error_rate_and_accuracy(cm):
    true_positive, false_positive, false_negative, true_negative = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    total = true_positive + false_positive + false_negative + true_negative
    error_rate = (false_positive + false_negative) / total
    accuracy = (true_positive + true_negative) / total
    return error_rate, accuracy


logistic_regression_error_rate, logistic_regression_accuracy = calculate_error_rate_and_accuracy(logistic_regression_cm)
decision_tree_error_rate, decision_tree_accuracy = calculate_error_rate_and_accuracy(decision_tree_cm)

# Calculate total_samples
total_samples = sum(sum(row) for row in logistic_regression_cm)

# Calculate all_died_accuracy
died_logistic_regression = logistic_regression_cm[1][1]
died_decision_tree = decision_tree_cm[1][1]
total_died = max(died_logistic_regression, died_decision_tree)
all_died_accuracy = total_died / total_samples

# Compare accuracies
print('10.5')
print("Logistic Regression - Error rate:", logistic_regression_error_rate, "Accuracy:", logistic_regression_accuracy)
print("Decision Tree - Error rate:", decision_tree_error_rate, "Accuracy:", decision_tree_accuracy)
print("All Died - Accuracy:", all_died_accuracy)
print('\n')

# 6
"""
The values of f00 (true negatives) and f11 (true positives) are obtained by applying the given threshold θ to the neural network's output (ˆyi) and comparing it with the true output (yi). Let's break down the calculations for each threshold value:

Given the table with the true output (yi) and neural network's output (ˆyi):

i	yi	ˆyi
1	1	0.6
2	0	0.4
3	1	0.5
4	1	0.1
5	0	0.1

For θ = 0.35:

Apply the threshold: ˆy1 = 1, ˆy2 = 1, ˆy3 = 1, ˆy4 = 0, ˆy5 = 0
Compare with true output (yi):
f00 (true negatives) = 1 (only the 5th example has both yi=0 and ˆyi=0)
f11 (true positives) = 2 (the 1st and 3rd examples have both yi=1 and ˆyi=1)

For θ = 0.45:

Apply the threshold: ˆy1 = 1, ˆy2 = 0, ˆy3 = 1, ˆy4 = 0, ˆy5 = 0
Compare with true output (yi):
f00 (true negatives) = 2 (the 2nd and 5th examples have both yi=0 and ˆyi=0)
f11 (true positives) = 2 (the 1st and 3rd examples have both yi=1 and ˆyi=1)

For θ = 0.55:

Apply the threshold: ˆy1 = 1, ˆy2 = 0, ˆy3 = 0, ˆy4 = 0, ˆy5 = 0
Compare with true output (yi):
f00 (true negatives) = 2 (the 2nd and 5th examples have both yi=0 and ˆyi=0)
f11 (true positives) = 1 (only the 1st example has both yi=1 and ˆyi=1)

For θ = 0.65:

Apply the threshold: ˆy1 = 0, ˆy2 = 0, ˆy3 = 0, ˆy4 = 0, ˆy5 = 0
Compare with true output (yi):
f00 (true negatives) = 2 (the 2nd and 5th examples have both yi=0 and ˆyi=0)
f11 (true positives) = 0 (no examples have both yi=1 and ˆyi=1)
These are the values used to calculate the accuracy for each threshold value.


n this problem, we want to find the threshold value (θ) that maximizes the accuracy when binarizing the neural network's output (ˆyi).

Given the threshold values θ = 0.35, 0.45, 0.55, and 0.65, we can count the number of true negatives (f00) and true positives (f11) for each threshold value. Then, we can compute the accuracy as (f00 + f11) / N, where N is the total number of observations.

For θ = 0.35:

f00 (true negatives) = 1
f11 (true positives) = 2
accuracy = (1 + 2) / 5 = 3 / 5 = 0.6
For θ = 0.45:

f00 (true negatives) = 2
f11 (true positives) = 2
accuracy = (2 + 2) / 5 = 4 / 5 = 0.8
For θ = 0.55:

f00 (true negatives) = 2
f11 (true positives) = 1
accuracy = (2 + 1) / 5 = 3 / 5 = 0.6
For θ = 0.65:

f00 (true negatives) = 2
f11 (true positives) = 0
accuracy = (2 + 0) / 5 = 2 / 5 = 0.4
Based on these calculations, the highest accuracy is obtained at θ = 0.45 with an accuracy of 0.8, making option B the correct answer.
"""

data = [
    [1, 0.6],
    [0, 0.4],
    [1, 0.5],
    [1, 0.1],
    [0, 0.1]
]

thresholds = [0.35, 0.45, 0.55, 0.65]


def calculate_accuracy(threshold):
    correct_predictions = 0
    for y, y_hat in data:
        binarized_y_hat = 1 if y_hat >= threshold else 0
        if binarized_y_hat == y:
            correct_predictions += 1
    return correct_predictions / len(data)


accuracies = [(threshold, calculate_accuracy(threshold)) for threshold in thresholds]
highest_accuracy = max(accuracies, key=lambda x: x[1])

print('10.6')
print("Highest Accuracy:", highest_accuracy)
print('\n')

# 7
"""
For Question 10.7, the correct answer is B: For datasets with very few observations, it is generally better to use leave-one-out cross-validation rather than 10-fold cross-validation.

Explanation: Leave-one-out cross-validation (LOOCV) makes better use of the limited data available by using all but one observation for training and validating on the single left-out observation. This is done for each observation in the dataset, resulting in a more reliable estimate of the model's performance. In contrast, 10-fold cross-validation would partition the dataset into 10 smaller subsets, which might not be representative of the overall data distribution when the dataset is small, leading to less accurate model performance estimates.

Now, let's discuss why the other options are incorrect:

A) 2-fold cross-validation is not the same as the hold-out method when 50% is held out. In 2-fold cross-validation, the data is split into two halves, but the model is trained and evaluated twice, once with each half as the test set. In the hold-out method, the model is trained on one half of the data and evaluated on the other half only once, without any repetition.

C) Two levels of cross-validation are not necessary to determine the optimal set of parameters for a model. One level of cross-validation (e.g., k-fold or leave-one-out) is often sufficient for parameter tuning and model selection. Nested cross-validation or two levels of cross-validation can be used to get a more accurate estimate of the model's performance, but it is not always necessary.

D) Leave-one-out cross-validation is not the most computationally efficient procedure, as it requires training the model N times, where N is the number of observations in the dataset. In contrast, k-fold cross-validation requires training the model k times, where k is often much smaller than N (e.g., k=10). While LOOCV has the advantage of making better use of limited data, it can be computationally expensive when the dataset is large.

10.7 The correct answer is B: For a very small dataset, it is better to use leave-one-out cross-validation as this 
will keep as much data for training as possible. 2-fold cross-validation is not the same as the hold-out method where 
50% is held out because two models are trained and evaluated on all the data by 2-fold cross-validation, 
whereas hold-out 50% only trains one model and evaluates the performance of this model on half the data. Only one 
level of cross-validation is needed for tuning model parameters. Two levels are used when quantifying the performance 
of the model with parameters selected. Leave-one-out cross-validation is computationally expensive since as many 
models as observations need to be trained. 
"""

# 8
"""
the right answer is B. Let me provide the updated explanation:

B: Forward selection will select a model with a higher misclassification rate on the test set than backward selection.
This is correct. Forward selection starts with no features and adds features iteratively. In this case, forward selection adds x2, then x2 and x4, then x2, x3, and x4, and finally x1, x2, x3, and x4. The test misclassification rate for this model is 25.

On the other hand, backward selection starts with all features (x1, x2, x3, and x4) and removes features iteratively. In this case, backward selection removes x2, forming the model with features x1, x3, and x4. The test misclassification rate for this model is 15. Thus, forward selection selects a model with a higher misclassification rate on the test set than backward selection.

The other options are incorrect for the following reasons:

A: Forward and backward selection will select the same set of features.
This is incorrect, as explained above.

C: Forward selection will select a model with a lower misclassification rate on the test set than backward selection.
This is incorrect because the test misclassification rate for the forward selection model is higher (25) compared to the backward selection model (15).

D: Forward selection will select the features x1, x3, x4.
This is incorrect because forward selection results in the features x1, x2, x3, and x4, not x1, x3, and x4.

E: Don't know.
This option is not correct as we have already determined the correct answer to be B.

In summary, the correct answer is B because forward selection leads to a higher misclassification rate on the test 
set compared to backward selection. The other options are incorrect because they either do not accurately describe 
the feature selection process or provide incorrect information about the results. 
"""

import itertools

test_results = {
    "x1": 70,
    "x2": 50,
    "x3": 76,
    "x4": 81,
    "x1,x2": 73,
    "x1,x3": 59,
    "x1,x4": 32,
    "x2,x3": 58,
    "x2,x4": 36,
    "x3,x4": 33,
    "x1,x2,x3": 40,
    "x1,x2,x4": 54,
    "x1,x3,x4": 15,
    "x2,x3,x4": 34,
    "x1,x2,x3,x4": 25
}

features = ["x1", "x2", "x3", "x4"]

# Forward Selection
min_error = float("inf")
best_features = None
selected_features = []

print('10.8')

for _ in range(len(features)):
    for feature in features:
        if feature not in selected_features:
            current_features = sorted(selected_features + [feature])  # Sort the features
            current_key = ",".join(current_features)
            error = test_results[current_key]
            if error < min_error:
                min_error = error
                best_features = current_features
    selected_features = best_features
    print(f"Forward selection step: {selected_features}, Test error: {min_error}")

# Backward Selection
min_error = float("inf")
best_features = None
selected_features = features

for _ in range(len(features)):
    for feature in selected_features:
        current_features = sorted([f for f in selected_features if f != feature])  # Sort the features
        current_key = ",".join(current_features)
        error = test_results[current_key]
        if error < min_error:
            min_error = error
            best_features = current_features
    selected_features = best_features
    print(f"Backward selection step: {selected_features}, Test error: {min_error}")

# Case A: Forward and backward selection will select the same set of features
forward_selection_features = selected_features.copy()
print(f"Forward selection final features: {forward_selection_features}")

# Reset selected_features and min_error for backward selection
selected_features = features
min_error = float("inf")

# Backward Selection
for _ in range(len(features)):
    for feature in selected_features:
        current_features = [f for f in selected_features if f != feature]
        current_key = ",".join(current_features)
        error = test_results[current_key]
        if error < min_error:
            min_error = error
            best_features = current_features
    selected_features = best_features

backward_selection_features = selected_features.copy()
print(f"Backward selection final features: {backward_selection_features}")

case_a = forward_selection_features == backward_selection_features
print(f"Case A: {case_a}")

# Case C: Forward selection will select a model with lower misclassification rate on the test set than backward selection
forward_error = test_results[",".join(forward_selection_features)]
backward_error = test_results[",".join(backward_selection_features)]

case_c = forward_error < backward_error
print(f"Case C: {case_c}")

# Case D: Forward selection will select the features x1, x3, x4
case_d = forward_selection_features == ["x1", "x3", "x4"]
print(f"Case D: {case_d}")
print('\n')

######################################
# 11
######################################
print("Section 11:")
# 1
"""
To calculate the p-value for McNemar's test, we first need to find the total number of observations where M1 is correct and M2 is wrong, and where M1 is wrong and M2 is correct. We can find these values by summing the respective columns in Table 11.2:

M1 correct, M2 wrong (M1/M 2): 8 + 15 + 5 = 28
M1 wrong, M2 correct (M 1/M2): 7 + 11 + 17 = 35

Now, we need to find the appropriate value in Table 11.3 using the smallest of these two values (28) as m and the total number of disagreements (28 + 35 = 63) as N. In Table 11.3, we find the corresponding value at m = 28 and N = 63: 0.225.

Since McNemar's test is a two-tailed test, we need to multiply this value by 2 to get the p-value:

p-value = 2 * 0.225 = 0.45

Thus, the correct answer is B: 0.45.
"""

from scipy.stats import binom

# Table 11.2 data
m1_m2 = [[86, 8, 7, 10],
         [65, 15, 11, 20],
         [79, 5, 17, 10]]

# Calculate total number of M1 correct, M2 wrong and M1 wrong, M2 correct
m1_correct_m2_wrong = sum(row[1] for row in m1_m2)
m1_wrong_m2_correct = sum(row[2] for row in m1_m2)

# Calculate the smallest value and total disagreements
m = min(m1_correct_m2_wrong, m1_wrong_m2_correct)
N = m1_correct_m2_wrong + m1_wrong_m2_correct

# Calculate the CDF using the binomial distribution
cdf_value = binom.cdf(m, N, 0.5)

# Calculate the p-value for McNemar's test (two-tailed)
p_value = 2 * cdf_value

print('11.1')
print("P-value:", p_value)
print('\n')

######################################
# 12
######################################
print("Section 12:")
# 1
"""
To find the error rate of the 3-nearest neighbor classifier using leave-one-out cross-validation, we need to check the three closest neighbors for each observation and determine whether they are correctly classified or not. Let's go through each observation:

O1 (small island): 3 closest neighbors are O3, O4, and O2 (all small islands) - Correctly classified
O2 (small island): 3 closest neighbors are O3, O4, and O1 (all small islands) - Correctly classified
O3 (small island): 3 closest neighbors are O1, O2, and O4 (all small islands) - Correctly classified
O4 (small island): 3 closest neighbors are O1, O3, and O2 (all small islands) - Correctly classified
O5 (large island): 3 closest neighbors are O7, O8, and O1 (two large islands and one small island) - misclassified
O6 (large island): 3 closest neighbors are O5, O7, and O8 (all large islands) - Correctly classified
O7 (large island): 3 closest neighbors are O5, O8, and O1 (two large islands and one small island) - misclassified
O8 (large island): 3 closest neighbors are O7, O5, and O3 (two large islands and one small island) - Correctly classified

two observations will be misclassified the error rate will be 2/8=1/4 -> B
"""
import numpy as np

# Pairwise Euclidean distance matrix
distance_matrix = np.array([
    [0, 2.39, 1.73, 0.96, 3.46, 4.07, 4.27, 5.11],
    [2.39, 0, 1.15, 1.76, 2.66, 5.36, 3.54, 4.79],
    [1.73, 1.15, 0, 1.52, 3.01, 4.66, 3.77, 4.90],
    [0.96, 1.76, 1.52, 0, 2.84, 4.25, 3.80, 4.74],
    [3.46, 2.66, 3.01, 2.84, 0, 4.88, 1.41, 2.96],
    [4.07, 5.36, 4.66, 4.25, 4.88, 0, 5.47, 5.16],
    [4.27, 3.54, 3.77, 3.80, 1.41, 5.47, 0, 2.88],
    [5.11, 4.79, 4.90, 4.74, 2.96, 5.16, 2.88, 0]
])

# True labels for the islands
labels = ['small', 'small', 'small', 'small', 'large', 'large', 'large', 'large']


def leave_one_out_knn(distance_matrix, labels, k):
    n = len(labels)
    errors = 0

    for i in range(n):
        distances = distance_matrix[i].copy()
        distances[i] = np.inf
        nearest_indices = np.argsort(distances)[:k]
        nearest_labels = [labels[index] for index in nearest_indices]
        prediction = max(set(nearest_labels), key=nearest_labels.count)

        if prediction != labels[i]:
            errors += 1

    error_rate = errors / n
    return error_rate


k = 3
error_rate = leave_one_out_knn(distance_matrix, labels, k)
print('12.1')
print(f"Error rate: {error_rate}")
print('\n')

# 2
"""
Upon closer examination of the figure, we can determine the correct values for k1, k2, k3, and k4 as follows:

For k1, the decision boundary is more flexible than k3, so k1 = 3.
For k2, the decision boundary classifies everything to the largest class, so k2 = 7.
For k3, the decision boundary has some flexibility but colors the ticks correctly, so k3 = 1.
For k4, by checking the left-most part of the k4-pane, we can see that k4 = 5.
So, the correct answer is:

D: k1 = 3, k2 = 7, k3 = 1, k4 = 5
"""

# 3
"""
To find the accuracy using leave-one-out cross-validation with k = 1 and city block distance, we will iterate through each observation and find its nearest neighbor using the city block distance. If the nearest neighbor belongs to the same class as the observation, we will count it as a correct classification. Finally, we will calculate the accuracy as the ratio of correct classifications to the total number of observations (N = 8).

Here's a step-by-step breakdown of the process:

For o1 (C1): Nearest neighbor is o2 (distance 4) - Same class (C1)
For o2 (C1): Nearest neighbor is o6 (distance 3) - Different class (C2)
For o3 (C1): Nearest neighbor is o7 (distance 4) - Different class (C2)
For o4 (C1): Nearest neighbor is o6 (distance 6) - Different class (C2)
For o5 (C2): Nearest neighbor is o1 (distance 5) - Different class (C1)
For o6 (C2): Nearest neighbor is o2 (distance 3) - Different class (C1)
For o7 (C2): Nearest neighbor is o3 (distance 4) - Different class (C1)
For o8 (C2): Nearest neighbor is o1 (distance 6) - Different class (C1)
From the above analysis, we see that only o1 was classified correctly, while the other seven observations were classified incorrectly. The accuracy is the ratio of correct classifications to the total number of observations:

Accuracy = (Number of correct classifications) / N
Accuracy = 1 / 8

So, the accuracy of the k = 1 nearest neighbor classifier using city block distance and leave-one-out cross-validation is 1/8 or 0.125 (12.5%).
"""
import numpy as np

# Cityblock distance matrix
distance_matrix = np.array([
    [0, 4, 7, 9, 5, 5, 5, 6],
    [4, 0, 7, 7, 7, 3, 7, 8],
    [7, 7, 0, 10, 6, 6, 4, 9],
    [9, 7, 10, 0, 8, 6, 10, 9],
    [5, 7, 6, 8, 0, 8, 6, 7],
    [5, 3, 6, 6, 8, 0, 8, 11],
    [5, 7, 4, 10, 6, 8, 0, 7],
    [6, 8, 9, 9, 7, 11, 7, 0]
])

# Class labels: 0 for C1 (blue), 1 for C2 (black)
class_labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])

# Perform leave-one-out cross-validation
correct_classifications = 0
for i in range(len(distance_matrix)):
    sorted_indices = np.argsort(distance_matrix[i])  # Indices of nearest neighbors in ascending order
    nearest_neighbor = sorted_indices[1]  # Index 0 is the observation itself, so we take the index at position 1

    if class_labels[i] == class_labels[nearest_neighbor]:
        correct_classifications += 1

accuracy = correct_classifications / len(distance_matrix)
print('12.3')
print("Accuracy:", accuracy)
print('\n')

######################################
# 13
######################################
print("Section 13:")
# 1
"""
We have the following probabilities:
P(CKD=1) = 9/15
P(CKD=0) = 6/15

P(RBC=1|CKD=1) = 2/9
P(PC=1|CKD=1) = 6/9
P(DM=1|CKD=1) = 5/9
P(CAD=1|CKD=1) = 1/9

P(RBC=1|CKD=0) = 1/6
P(PC=1|CKD=0) = 1/6
P(DM=1|CKD=0) = 1/6
P(CAD=1|CKD=0) = 1/6

Now, we apply the Naive Bayes formula:

numerator = P(CKD=1) * P(RBC=1|CKD=1) * P(PC=1|CKD=1) * P(DM=1|CKD=1) * P(CAD=1|CKD=1)
denominator = numerator + (P(CKD=0) * P(RBC=1|CKD=0) * P(PC=1|CKD=0) * P(DM=1|CKD=0) * P(CAD=1|CKD=0))

The result is:

P(CKD=1|RBC=1, PC=1, DM=1, CAD=1) = numerator / denominator = 1

So, the correct answer is: 
0.9614

To compute the probability P(CKD=1|RBC=1, PC=1, DM=1, CAD=1) using the Naïve Bayes classifier, we first need to calculate the individual probabilities for each attribute given CKD=1 or CKD=0. Then, we will use the Bayes theorem to find the final probability.

First, let's find the prior probabilities for CKD=1 and CKD=0:
P(CKD=1) = 9/15
P(CKD=0) = 6/15

Now, let's find the conditional probabilities for each attribute given CKD=1 or CKD=0:

RBC=1:
P(RBC=1|CKD=1) = 2/9
P(RBC=1|CKD=0) = 1/6

PC=1:
P(PC=1|CKD=1) = 6/9
P(PC=1|CKD=0) = 1/6

DM=1:
P(DM=1|CKD=1) = 4/9
P(DM=1|CKD=0) = 1/6

CAD=1:
P(CAD=1|CKD=1) = 1/9
P(CAD=1|CKD=0) = 1/6

Now, we can use the Naïve Bayes formula to calculate the probability:

P(CKD=1|RBC=1, PC=1, DM=1, CAD=1) = P(CKD=1) * P(RBC=1|CKD=1) * P(PC=1|CKD=1) * P(DM=1|CKD=1) * P(CAD=1|CKD=1) / P(RBC=1, PC=1, DM=1, CAD=1)

To find the denominator P(RBC=1, PC=1, DM=1, CAD=1), we can use the Law of Total Probability:

P(RBC=1, PC=1, DM=1, CAD=1) = P(CKD=1) * P(RBC=1|CKD=1) * P(PC=1|CKD=1) * P(DM=1|CKD=1) * P(CAD=1|CKD=1) + P(CKD=0) * P(RBC=1|CKD=0) * P(PC=1|CKD=0) * P(DM=1|CKD=0) * P(CAD=1|CKD=0)

Plugging in the values, we get:

P(RBC=1, PC=1, DM=1, CAD=1) = (9/15) * (2/9) * (6/9) * (4/9) * (1/9) + (6/15) * (1/6) * (1/6) * (1/6) * (1/6)

P(RBC=1, PC=1, DM=1, CAD=1) ≈ 0.000582

Finally, we can calculate P(CKD=1|RBC=1, PC=1, DM=1, CAD=1):

P(CKD=1|RBC=1, PC=1, DM=1, CAD=1) ≈ ((9/15) * (2/9) * (6/9) * (4/9) * (1/9)) / 0.000582
P(CKD=1|RBC=1, PC=1, DM=1, CAD=1) ≈ 0.969

So, according to the Naïve Bayes classifier, the probability that a person with attributes RBC=1, PC=1, DM=1, and CAD=1 has chronic kidney disease (CKD=1) is approximately 0.969 or 96.9%.
"""

# Probabilities
P_CKD_1 = 9 / 15
P_CKD_0 = 6 / 15

P_RBC_1_CKD_1 = 2 / 9
P_PC_1_CKD_1 = 6 / 9
P_DM_1_CKD_1 = 5 / 9
P_CAD_1_CKD_1 = 1 / 9

P_RBC_1_CKD_0 = 1 / 6
P_PC_1_CKD_0 = 1 / 6
P_DM_1_CKD_0 = 1 / 6
P_CAD_1_CKD_0 = 1 / 6

# Naive Bayes formula
numerator = P_CKD_1 * P_RBC_1_CKD_1 * P_PC_1_CKD_1 * P_DM_1_CKD_1 * P_CAD_1_CKD_1
denominator = numerator + (P_CKD_0 * P_RBC_1_CKD_0 * P_PC_1_CKD_0 * P_DM_1_CKD_0 * P_CAD_1_CKD_0)

result = numerator / denominator
print("13.1")
print("P(CKD=1|RBC=1, PC=1, DM=1, CAD=1) =", result)
print("\n")

# or

import pandas as pd

data = {'RBC': [0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
        'PC': [0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        'DM': [1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0],
        'CAD': [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
        'CKD': [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]}
df = pd.DataFrame(data)

ckd_1 = df[df['CKD'] == 1]
ckd_0 = df[df['CKD'] == 0]

P_CKD_1 = len(ckd_1) / len(df)
P_CKD_0 = len(ckd_0) / len(df)

P_RBC_1_given_CKD_1 = sum(ckd_1['RBC'] == 1) / len(ckd_1)
P_PC_1_given_CKD_1 = sum(ckd_1['PC'] == 1) / len(ckd_1)
P_DM_1_given_CKD_1 = sum(ckd_1['DM'] == 1) / len(ckd_1)
P_CAD_1_given_CKD_1 = sum(ckd_1['CAD'] == 1) / len(ckd_1)

P_RBC_1_given_CKD_0 = sum(ckd_0['RBC'] == 1) / len(ckd_0)
P_PC_1_given_CKD_0 = sum(ckd_0['PC'] == 1) / len(ckd_0)
P_DM_1_given_CKD_0 = sum(ckd_0['DM'] == 1) / len(ckd_0)
P_CAD_1_given_CKD_0 = sum(ckd_0['CAD'] == 1) / len(ckd_0)

numerator = P_CKD_1 * P_RBC_1_given_CKD_1 * P_PC_1_given_CKD_1 * P_DM_1_given_CKD_1 * P_CAD_1_given_CKD_1
denominator = numerator + (
        P_CKD_0 * P_RBC_1_given_CKD_0 * P_PC_1_given_CKD_0 * P_DM_1_given_CKD_0 * P_CAD_1_given_CKD_0)

P_CKD_1_given_RBC_1_PC_1_DM_1_CAD_1 = numerator / denominator
print("13.1")
print("P(CKD=1|RBC=1, PC=1, DM=1, CAD=1) =", P_CKD_1_given_RBC_1_PC_1_DM_1_CAD_1)
print("\n")

# 2
"""
To answer this question using a Bayes classifier without assuming independence between the attributes RBC, PC, and DM, we need to compute the joint probabilities directly from the data. We will follow these steps:

Calculate P(CKD=1), P(RBC=1|CKD=1), P(PC=1|CKD=1), P(DM=1|CKD=1), and their corresponding probabilities for CKD=0.
Calculate the joint probability P(RBC=1, PC=1, DM=1, CKD=1) and P(RBC=1, PC=1, DM=1, CKD=0).
Use the Bayes theorem to find P(CKD=1|RBC=1, PC=1, DM=1).
Step 1: We have already computed these probabilities in the previous question:

P(CKD=1) = 9/15
P(CKD=0) = 6/15
P(RBC=1|CKD=1) = 2/9
P(RBC=1|CKD=0) = 1/6
P(PC=1|CKD=1) = 6/9
P(PC=1|CKD=0) = 1/6
P(DM=1|CKD=1) = 4/9
P(DM=1|CKD=0) = 1/6
Step 2: We need to find the number of occurrences where RBC=1, PC=1, DM=1, and CKD=1:

There is only 1 occurrence in the table with these conditions: O6.

So, P(RBC=1, PC=1, DM=1, CKD=1) = 1/15

For RBC=1, PC=1, DM=1, and CKD=0, there are no occurrences in the table.

So, P(RBC=1, PC=1, DM=1, CKD=0) = 0

Step 3: Use Bayes theorem to find P(CKD=1|RBC=1, PC=1, DM=1):

P(CKD=1|RBC=1, PC=1, DM=1) = P(RBC=1, PC=1, DM=1, CKD=1) / (P(RBC=1, PC=1, DM=1, CKD=1) + P(RBC=1, PC=1, DM=1, CKD=0))

P(CKD=1|RBC=1, PC=1, DM=1) = (1/15) / ((1/15) + 0) = 1

So, the probability P(CKD=1|RBC=1, PC=1, DM=1) according to the Bayes classifier is 100% (Option D).
"""
import numpy as np

# Table 13.2 data
data = np.array([
    [0, 0, 0, 0, 1, 0, 0],
    [0, 1, 1, 1, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 1, 0, 1],
    [0, 1, 1, 1, 1, 0, 0],
    [1, 1, 1, 1, 1, 0, 0],
    [1, 1, 1, 1, 1, 0, 1],
    [0, 1, 1, 1, 1, 1, 1],
    [0, 1, 0, 1, 0, 0, 0],
    [1, 1, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0]
])

# CKD (chronic kidney disease) labels
ckd_labels = np.array([1] * 9 + [0] * 6)


# Calculate joint probability
def joint_probability(rbc, pc, dm, ckd):
    # Filter the data by the CKD label
    data_filtered = data[ckd_labels == ckd]

    # Count the occurrences where RBC, PC, and DM have the specified values
    count = np.sum((data_filtered[:, 0] == rbc) & (data_filtered[:, 1] == pc) & (data_filtered[:, 4] == dm))

    # Calculate the joint probability
    prob = count / len(data)
    return prob


# Calculate P(CKD=1|RBC=1, PC=1, DM=1) using the Bayes theorem
numerator = joint_probability(1, 1, 1, 1)
denominator = joint_probability(1, 1, 1, 1) + joint_probability(1, 1, 1, 0)
bayes_result = numerator / denominator

print('13.2')
print("P(CKD=1|RBC=1, PC=1, DM=1) =", bayes_result)
print('\n')

# 3
"""
We will apply a Naïve Bayes classifier to predict the probability of survival given the attributes YAY=1, OAY=1, and PAY=1. To do this, we will follow these steps:

Calculate the probabilities P(S), P(YAY=1|S), P(OAY=1|S), and P(PAY=1|S) and their corresponding probabilities for not surviving (NS).
Use the Naïve Bayes formula to compute P(S|YAY=1, OAY=1, PAY=1).
Step 1: Calculate the probabilities:

P(S) = 5/10 (5 subjects survived out of 10)
P(NS) = 5/10 (5 subjects did not survive out of 10)
P(YAY=1|S) = 2/5 (2 out of 5 survivors have YAY=1)
P(YAY=1|NS) = 1/5 (1 out of 5 non-survivors have YAY=1)
P(OAY=1|S) = 4/5 (4 out of 5 survivors have OAY=1)
P(OAY=1|NS) = 4/5 (4 out of 5 non-survivors have OAY=1)
P(PAY=1|S) = 4/5 (4 out of 5 survivors have PAY=1)
P(PAY=1|NS) = 4/5 (4 out of 5 non-survivors have PAY=1)
Step 2: Use the Naïve Bayes formula:

P(S|YAY=1, OAY=1, PAY=1) = P(S) * P(YAY=1|S) * P(OAY=1|S) * P(PAY=1|S) / P(YAY=1, OAY=1, PAY=1)

To compute P(YAY=1, OAY=1, PAY=1), we have to calculate:

P(YAY=1, OAY=1, PAY=1) = P(S) * P(YAY=1|S) * P(OAY=1|S) * P(PAY=1|S) + P(NS) * P(YAY=1|NS) * P(OAY=1|NS) * P(PAY=1|NS)

P(YAY=1, OAY=1, PAY=1) = (5/10) * (2/5) * (4/5) * (4/5) + (5/10) * (1/5) * (4/5) * (4/5)

P(YAY=1, OAY=1, PAY=1) = 0.112

Now we can calculate P(S|YAY=1, OAY=1, PAY=1):

P(S|YAY=1, OAY=1, PAY=1) = ((5/10) * (2/5) * (4/5) * (4/5)) / 0.112

P(S|YAY=1, OAY=1, PAY=1) ≈ 0.72 = D

So, according to the Naïve Bayes classifier, the probability that a subject with attributes YAY=1, OAY=1, and PAY=1 survived is approximately 0.72 or 72%.
"""

import numpy as np

# Table 13.3 data
data = np.array([
    [1, 0, 1, 0, 1, 0],
    [1, 0, 1, 0, 0, 1],
    [0, 1, 0, 1, 1, 0],
    [0, 1, 1, 0, 1, 0],
    [0, 1, 1, 0, 1, 0],
    [0, 1, 1, 0, 1, 0],
    [0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 1],
    [0, 1, 1, 0, 1, 0],
    [0, 1, 1, 0, 1, 0]
])

# Survival labels
survival_labels = np.array([1] * 5 + [0] * 5)


# Calculate conditional probabilities
def conditional_probability(attr_idx, attr_val, survival):
    data_filtered = data[survival_labels == survival]
    count = np.sum(data_filtered[:, attr_idx] == attr_val)
    prob = count / len(data_filtered)
    return prob


# Calculate probabilities
p_s = 5 / 10
p_ns = 1 - p_s
p_yay_given_s = conditional_probability(0, 1, 1)
p_yay_given_ns = conditional_probability(0, 1, 0)
p_oay_given_s = conditional_probability(2, 1, 1)
p_oay_given_ns = conditional_probability(2, 1, 0)
p_pay_given_s = conditional_probability(4, 1, 1)
p_pay_given_ns = conditional_probability(4, 1, 0)

# Calculate joint probabilities
joint_s = p_s * p_yay_given_s * p_oay_given_s * p_pay_given_s
joint_ns = p_ns * p_yay_given_ns * p_oay_given_ns * p_pay_given_ns

# Calculate P(S|YAY=1, OAY=1, PAY=1) using the Naïve Bayes formula
p_s_given_yay_oay_pay = joint_s / (joint_s + joint_ns)

print('13.3')
print("P(S|YAY=1, OAY=1, PAY=1) =", p_s_given_yay_oay_pay)
print('\n')

# 4
"""
Given:

p(f1=0|C1) = 3/5
p(f2=1|C1) = 1
p(f1=0|C2) = 1/5
p(f2=1|C2) = 3/5
p(C1) = p(C2) = 1/2

These probabilities are derived from the data in Table 13.4, which contains 10 observations (s1 to s10) with 6 binary features (f1 to f6). The table is split into two categories: class C1 (observations s1 to s5) and class C2 (observations s6 to s10).

To calculate the given probabilities, we need to count occurrences of each feature value (0 or 1) in each class.

p(f1=0|C1): The probability of f1 being 0 given that the observation is in class C1.
From the data, there are 5 observations in class C1. Out of these, f1 is 0 for 3 observations (s1, s2, and s5). So, p(f1=0|C1) = 3/5.

p(f2=1|C1): The probability of f2 being 1 given that the observation is in class C1.
In class C1, f2 is 1 for all 5 observations. So, p(f2=1|C1) = 5/5 = 1.

p(f1=0|C2): The probability of f1 being 0 given that the observation is in class C2.
In class C2, there are 5 observations, and f1 is 0 for only 1 observation (s6). So, p(f1=0|C2) = 1/5.

p(f2=1|C2): The probability of f2 being 1 given that the observation is in class C2.
In class C2, f2 is 1 for 3 out of the 5 observations (s6, s7, and s10). So, p(f2=1|C2) = 3/5.

p(C1) = p(C2) = 1/2: There are 10 observations in total, with 5 in class C1 and 5 in class C2. The class priors for both classes are equal, so p(C1) = p(C2) = 5/10 = 1/2.

We want to find p_NB(C1|f1=0, f2=1):

p_NB(C1|f1=0, f2=1) = p(f1=0|C1) * p(f2=1|C1) * p(C1) / (p(f1=0|C1) * p(f2=1|C1) * p(C1) + p(f1=0|C2) * p(f2=1|C2) * p(C2))

p_NB(C1|f1=0, f2=1) = ((3/5) * 1 * (1/2)) / ((3/5) * 1 * (1/2) + (1/5) * (3/5) * (1/2))

p_NB(C1|f1=0, f2=1) = (3/5) / (3/5 + (1/5) * (3/5))

p_NB(C1|f1=0, f2=1) = (3/5) / (3/5 + 3/25)

p_NB(C1|f1=0, f2=1) = 5/6

So, according to the Naïve Bayes classifier, 
the probability that an observation with 
f1=0 and f2=1 belongs to class C1 is 5/6, which is approximately 0.83 or 83%.
"""

# Probabilities
p_f1_given_C1 = 3 / 5
p_f2_given_C1 = 1
p_f1_given_C2 = 1 / 5
p_f2_given_C2 = 3 / 5
p_C1 = 1 / 2
p_C2 = 1 / 2

# Naïve Bayes estimate
numerator = p_f1_given_C1 * p_f2_given_C1 * p_C1
denominator = (p_f1_given_C1 * p_f2_given_C1 * p_C1) + (p_f1_given_C2 * p_f2_given_C2 * p_C2)

p_NB_C1_given_f1_f2 = numerator / denominator

print('13.4')
print("P(C1|f1=0, f2=1) =", p_NB_C1_given_f1_f2)
print('\n')

######################################
# 14
######################################
print("Section 14:")
# 1
"""
14.1. Question 1: The incorrect statement is:

B. Linear regression where the inputs are transformed can only model linear relations between the original untransformed inputs and the outputs.

Explanation: Transforming inputs can actually help linear regression model non-linear relationships between the
original untransformed inputs and the outputs. For example, using polynomial features or logarithmic transformations
can capture non-linear patterns in the data.

Other statements:

A. In regularized least squares regression the aim is to reduce the model’s variance without introducing too much bias.

Explanation: This is correct because regularization adds a penalty term to the model, discouraging it from fitting
the noise in the training data. The result is a model that has reduced variance but doesn't introduce too much bias.

C. To investigate what attribute transformations may be relevant to consider, it is useful to plot each attribute versus the residuals.

Explanation: This is correct because plotting attributes against residuals can help identify if there is any
structure or pattern in the residuals, which might indicate that a transformation of the attribute could be
beneficial to capture the relationship with the output better.

D. Forward selection can be used both for regression and classification problems.

Explanation: This is correct because forward selection is a feature selection technique that can be applied to both
regression and classification problems. The technique iteratively adds one feature at a time to the model,
choosing the one that provides the most significant improvement in model performance. """
# 2
"""
14.2. Question 2: The option that will typically increase the amount of over-fitting is:

B. Reduce the amount of training data.

Explanation: Reducing the amount of training data will make the model more likely to overfit, as it will have less 
information to learn from and will be more prone to fitting the noise in the data. 

Other options:

A. Reduce the number of attributes.

Explanation: Reducing the number of attributes will generally reduce overfitting, as it simplifies the model by 
removing irrelevant or redundant features. 

C. Select a less complex model.

Explanation: Selecting a less complex model will generally reduce overfitting, as simpler models have less capacity 
to fit the noise in the data. 

D. Add model regularization.

Explanation: Adding model regularization will generally reduce overfitting, as it adds a penalty term to the model 
that discourages it from fitting the noise in the training data. """

######################################
# 15
######################################
print("Section 15:")
# 1
"""
The right answer is C
C. An artificial neural network with linear transfer functions (q(t) = t) can be written in terms of a linear regression model.

Explanation: When the transfer function of an artificial neural network is linear, the output of the model is a 
linear combination of the input features, which is the same structure as a linear regression model. The other 
statements are either incorrect or not relevant to regression problems. 

A. In regularized least squares regression the aim is to introduce more variance by reducing substantially the model’s bias.

Incorrect: The aim of regularized least squares regression is to reduce overfitting by introducing a regularization 
term that penalizes large weights. This usually reduces the model's variance while potentially increasing its bias, 
leading to a better balance between bias and variance. 

B. In least squares regularized regression the regularization strength λ is chosen to be the value of λ that 
minimizes the term λw⊤w. 

Incorrect: The regularization strength λ is chosen to balance the trade-off between fitting the training data (
minimizing the least squares error) and penalizing large weights (minimizing the regularization term). The value of λ 
is not selected to minimize λw⊤w alone, but to minimize the overall objective function, which includes both the least 
squares error term and the regularization term. 

D. For regression problems backward or forward selection can be used to define which part of the output that is 
relevant for modeling. 

Incorrect: Backward and forward selection are feature selection methods used to determine which input features are 
most relevant for modeling, not which part of the output. These methods help in selecting a subset of input features 
that contribute the most to the model's predictive performance while avoiding overfitting. """

# 2
"""
Let's calculate the output of the feedforward neural network with the given weights and input values.

Input values: x1 = 1, x2 = 2

Activation function: f(x) = {x if x > 0, 0 otherwise}

Weights:

w31 = 0.5
w41 = 0.4
w32 = -0.4
w42 = 0
w53 = -0.4
w54 = 0.1

Calculations for hidden layer:

n3 = f(w31 * x1 + w32 * x2) = f(0.5 * 1 - 0.4 * 2) = f(-0.3)
n4 = f(w41 * x1 + w42 * x2) = f(0.4 * 1 + 0 * 2) = f(0.4)

Since f(x) = 0 for x ≤ 0:

n3 = 0
n4 = 0.4

Calculations for output layer:

n5 = f(w53 * n3 + w54 * n4) = f(-0.4 * 0 + 0.1 * 0.4) = f(0.04)

Since 0.04 > 0:

n5 = 0.04

The output of the neural network is 0.04.
"""


def activation_function(x):
    return x if x > 0 else 0


# Input values
x1 = 1
x2 = 2

# Weights
w31 = 0.5
w41 = 0.4
w32 = -0.4
w42 = 0
w53 = -0.4
w54 = 0.1

# Calculations for hidden layer
n3 = activation_function(w31 * x1 + w32 * x2)
n4 = activation_function(w41 * x1 + w42 * x2)

# Calculations for output layer
n5 = activation_function(w53 * n3 + w54 * n4)

print("15.2")
print("Output of the neural network:", n5)
print("\n")

# 3
"""
The correct statement is D:

Classifier 1 is the logistic regression model, Classifier 2 is the 1-nearest neighbor, Classifier 3 is the decision 
tree, and classifier 4 is the artificial neural network. 

Here's why:

Logistic regression model: The decision boundary is linear. Classifier 1 has a straight line separating the two 
classes, which indicates a linear decision boundary typical for logistic regression. 

1-nearest neighbor: The decision boundary is more irregular and sensitive to individual data points. Classifier 2 has an irregular and complex 
boundary, which is characteristic of a 1-nearest neighbor classifier. 

Decision tree: The decision boundary is axis-aligned and piecewise constant. Classifier 3 has a rectangular shape, which is typical for decision trees since 
they create axis-aligned splits in the feature space. 

Artificial neural network: The decision boundary can be complex 
and non-linear. Classifier 4 has a smooth, curved decision boundary, which suggests that it is an artificial neural 
network with hidden units that can learn complex relationships between features. 
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# Create a toy dataset
X, y = make_classification(n_samples=200, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1,
                           random_state=42)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the classifiers
classifiers = [
    ("Logistic Regression", LogisticRegression()),
    ("1-NN", KNeighborsClassifier(n_neighbors=1)),
    ("Decision Tree", DecisionTreeClassifier(max_depth=3)),
    ("Neural Network", MLPClassifier(hidden_layer_sizes=(4,), max_iter=1000, random_state=42))
]


# Define function to plot decision boundaries
def plot_decision_boundary(clf, X, y, ax):
    h = .02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.8)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', s=100)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    return scatter


# Plot the decision boundaries
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
for (name, clf), ax in zip(classifiers, axes.ravel()):
    clf.fit(X_train, y_train)
    scatter = plot_decision_boundary(clf, X, y, ax)
    ax.set_title(name)
plt.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
plt.title("15.3")
plt.show()

# 4
"""
Based on the characteristics of the classifiers and their decision boundaries, the correct answer is:

C: P1 is LREG, P2 is TREE, P3 is 1NN, P4 is NNET.

Here's a brief explanation for each classifier's decision boundary:

P1 (LREG): Logistic regression usually produces linear decision boundaries. P1 exhibits a straight line separating the two classes, which is consistent with logistic regression.

P2 (TREE): Decision trees often generate axis-aligned, rectangular decision boundaries. In P2, we can see that the boundary has a rectangular shape, consistent with a decision tree.

P3 (1NN): The 1-nearest neighbor classifier typically creates a more complex and irregular decision boundary. P3 shows a non-linear, irregular boundary that closely follows the shape of the data points, which is consistent with a 1-nearest neighbor classifier.

P4 (NNET): Artificial neural networks can create more flexible, non-linear decision boundaries. In P4, we see a smooth, curved boundary that separates the two classes, consistent with an artificial neural network with a few hidden units.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# Create a toy dataset
X, y = make_classification(n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the classifiers
classifiers = [
    ('LREG', LogisticRegression(random_state=42)),
    ('TREE', DecisionTreeClassifier(random_state=42)),
    ('1NN', KNeighborsClassifier(n_neighbors=1)),
    ('NNET', MLPClassifier(hidden_layer_sizes=(4,), random_state=42))
]


# Function to plot decision boundaries
def plot_decision_boundary(ax, classifier, X, y, title):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.8)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', s=50, cmap=plt.cm.coolwarm)
    ax.set_title(title)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    return scatter


# Train the classifiers and plot their decision boundaries
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

for (name, clf), ax in zip(classifiers, axs.ravel()):
    clf.fit(X_train, y_train)
    plot_decision_boundary(ax, clf, X, y, name)

plt.title("15.4")
plt.show()

######################################
# 16
######################################
print("Section 16:")
# 1
"""
To determine which ROC curve corresponds to using the feature x1, we need to calculate the true positive rate (TPR) and false positive rate (FPR) for x1. We can do this by examining the data in Table 16.1.

First, we count the number of subjects with inflammation of the urinary bladder (y = 1) and without inflammation (y = 0):

y = 1: 4 subjects (P1, P5, P9, P10)
y = 0: 10 subjects (P2, P3, P4, P6, P7, P8, P11, P12, P13, P14)
Next, we look at the x1 feature:

True positives (TP): Subjects with x1 = 1 and y = 1 -> 3 subjects (P1, P5, P9)
False positives (FP): Subjects with x1 = 1 and y = 0 -> 3 subjects (P3, P7, P12)
True negatives (TN): Subjects with x1 = 0 and y = 0 -> 7 subjects (P2, P4, P6, P8, P11, P13, P14)
False negatives (FN): Subjects with x1 = 0 and y = 1 -> 1 subject (P10)
Now we calculate the TPR and FPR:
TP: 3
FP: 3
TN: 8
FN: 0

TPR = TP / (TP + FN) = 3 / (3 + 0) = 3/3 = 1
FPR = FP / (FP + TN) = 3 / (3 + 8) = 3/11 = 0.27
Since we don't have the actual ROC curves in Fig. 16.9, we can't directly determine which curve corresponds to using the feature x1 based on TPR and FPR. However, the AUC values give us some information about the performance of each curve. In general, an AUC closer to 1 indicates better classification performance.

Given the calculated TPR and FPR, we can make an educated guess that the curve with AUC=0.864 (option B) might correspond to using the feature x1. This is because the TPR is relatively high and FPR is relatively low, which would result in a higher AUC. However, without the actual ROC curves or more information, we can't be certain about the answer.
"""
import pandas as pd

data = {'x1': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0],
        'y': [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0]}

df = pd.DataFrame(data)

# Count the number of subjects with and without inflammation
y_1 = sum(df['y'] == 1)
y_0 = sum(df['y'] == 0)

# Calculate TP, FP, TN, FN
TP = sum((df['x1'] == 1) & (df['y'] == 1))
FP = sum((df['x1'] == 1) & (df['y'] == 0))
TN = sum((df['x1'] == 0) & (df['y'] == 0))
FN = sum((df['x1'] == 0) & (df['y'] == 1))

print('16.2')
print(f'TP: {TP}')
print(f'FP: {FP}')
print(f'TN: {TN}')
print(f'FN: {FN}')

# Calculate TPR and FPR
TPR = TP / (TP + FN)
FPR = FP / (FP + TN)

print(f'TPR: {TPR}, FPR: {FPR}')

# Educated guess based on AUC values
auc_values = [0.636, 0.864, 0.909, 1.000]
best_auc_index = auc_values.index(max(auc_values))
print(
    f"Based on the calculated TPR and FPR, the curve with AUC={auc_values[best_auc_index]} might correspond to using the feature x1.")

import numpy as np

data = np.array([
    [1, 1, 1, 1, 0, 1],
    [0, 0, 0, 0, 0, 0],
    [1, 1, 0, 1, 0, 0],
    [0, 1, 1, 0, 1, 0],
    [1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0],
    [1, 1, 0, 1, 0, 0],
    [0, 1, 1, 0, 1, 0],
    [1, 1, 1, 1, 0, 1],
    [0, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0],
    [1, 1, 0, 1, 0, 0],
    [0, 1, 1, 0, 1, 0],
    [0, 1, 1, 0, 1, 0],
])

x1 = data[:, 0]
y = data[:, -1]

TP = np.sum((x1 == 1) & (y == 1))
FP = np.sum((x1 == 1) & (y == 0))
TN = np.sum((x1 == 0) & (y == 0))
FN = np.sum((x1 == 0) & (y == 1))

TPR = TP / (TP + FN)
FPR = FP / (FP + TN)

print(f"TPR: {TPR}, FPR: {FPR}")

import numpy as np
from sklearn.metrics import roc_auc_score

data = np.array([
    # [x1, y]
    [1, 1],
    [0, 0],
    [1, 0],
    [0, 0],
    [1, 1],
    [0, 0],
    [1, 0],
    [0, 0],
    [1, 1],
    [0, 0],
    [0, 0],
    [1, 0],
    [0, 0],
    [0, 0]
])

x1 = data[:, 0]
y = data[:, 1]

auc = roc_auc_score(y, x1)
print("AUC:", auc)
print('\n')

# 2
"""
The right output in code, where theory has wrong number but correct method

In order to determine which ROC curve from Figure 16.10 corresponds to the logistic regression classifier in Figure 16.11, follow these theoretical steps:

Analyze the output of the logistic regression classifier and create a contingency table with the predicted probabilities and true labels.

Determine the thresholds for the predicted probabilities. In this case, you can select the unique predicted probabilities: -0.4, -0.1, 0.3, 0.6, 0.7, 0.8, and 0.9.

Calculate the true positive rate (TPR) and false positive rate (FPR) for each threshold.

Create the ROC curve by plotting the TPR against FPR for each threshold.

Compare the generated ROC curve with the ROC curves given in Figure 16.10.

Choose the ROC curve from Figure 16.10 that matches the generated ROC curve.

Now, let's analyze the output of the logistic regression classifier:
(x8, y) (-0.4, 0) (-0.1, 0) (0.3, 1) (0.6, 1) (0.7, 0) (0.8, 1) (0.9, 0)

Calculate the TPR and FPR for each threshold:

Threshold	TPR	        FPR
-0.4	    1/3 (0.33)	2/4 (0.5)
-0.1	    1/3 (0.33)	1/4 (0.25)
0.3	2/3     (0.67)	    1/4 (0.25)
0.6	3/3     (1)	1/4     (0.25)
0.7	3/3     (1)	2/4     (0.5)
0.8	3/3     (1)	3/4     (0.75)
0.9	3/3     (1)	4/4     (1)

Plot the TPR and FPR for each threshold:
(0.25, 0.67) (0.25, 1) (0.5, 0.33) (0.5, 1) (0.75, 1) (1, 1)

Compare the generated ROC curve with the ROC curves given in Figure 16.10:

ROC curve 1: (0, 0.5) (0.65, 0.75) (1, 1)
ROC curve 2: (0, 0.25) (0.37, 0.5) (1, 1)
ROC curve 3: (0.25, 0.37) (0.5, 1) (1, 1)
ROC curve 4: (0.5, 0.5) (1, 1)

By comparing the generated ROC curve with the given ROC curves, we can observe that the generated ROC curve is most similar to ROC curve 3 from Figure 16.10. Therefore, the answer is C (ROC curve 3).


-----
To calculate the True Positive Rate (TPR) and False Positive Rate (FPR) for each threshold, follow these steps:

Sort the predicted probabilities in ascending order along with their corresponding true labels.
For each unique predicted probability value, set it as a threshold.
Calculate the TPR and FPR for each threshold.
Here's a step-by-step breakdown for calculating the TPR and FPR for each threshold:

1- Sort the predicted probabilities along with their true labels:
(-0.4, 0) (-0.1, 0) (0.3, 1) (0.6, 1) (0.7, 0) (0.8, 1) (0.9, 0)

2- Set each unique predicted probability value as a threshold:
-0.4, -0.1, 0.3, 0.6, 0.7, 0.8, 0.9
 
3- Calculate the TPR and FPR for each threshold:
For each threshold, count the True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN) when comparing the predicted probabilities with their true labels.

True Positive (TP): predicted probability >= threshold and true label is 1
False Positive (FP): predicted probability >= threshold and true label is 0
True Negative (TN): predicted probability < threshold and true label is 0
False Negative (FN): predicted probability < threshold and true label is 1
Then, calculate the TPR and FPR using these counts:

TPR = TP / (TP + FN)
FPR = FP / (FP + TN)
For example, let's calculate the TPR and FPR for the threshold of 0.3:

TP: 2 (0.3, 1) and (0.6, 1)
FP: 1 (0.7, 0)
TN: 3 (-0.4, 0), (-0.1, 0), and (0.9, 0)
FN: 1 (0.8, 1)
Now calculate TPR and FPR:

TPR = TP / (TP + FN) = 2 / (2 + 1) = 2/3 ≈ 0.67
FPR = FP / (FP + TN) = 1 / (1 + 3) = 1/4 = 0.25
Repeat this process for all the thresholds and then plot the TPR against FPR for each threshold to create the ROC curve.

Certainly! Let's calculate the TPR and FPR for all the thresholds and plot the TPR against FPR for each threshold to create the ROC curve:

Threshold = -0.4:
TP = 3, FP = 4, TN = 0, FN = 0
TPR = 3 / (3 + 0) = 1
FPR = 4 / (4 + 0) = 1
Threshold = -0.1:
TP = 3, FP = 3, TN = 1, FN = 0
TPR = 3 / (3 + 0) = 1
FPR = 3 / (3 + 1) = 0.75
Threshold = 0.3 (we already calculated this one):
TP = 2, FP = 1, TN = 3, FN = 1
TPR = 2 / (2 + 1) ≈ 0.67
FPR = 1 / (1 + 3) = 0.25
Threshold = 0.6:
TP = 1, FP = 1, TN = 3, FN = 2
TPR = 1 / (1 + 2) ≈ 0.33
FPR = 1 / (1 + 3) = 0.25
Threshold = 0.7:
TP = 1, FP = 0, TN = 4, FN = 2
TPR = 1 / (1 + 2) ≈ 0.33
FPR = 0 / (0 + 4) = 0
Threshold = 0.8:
TP = 0, FP = 0, TN = 4, FN = 3
TPR = 0 / (0 + 3) = 0
FPR = 0 / (0 + 4) = 0
Threshold = 0.9:
TP = 0, FP = 0, TN = 4, FN = 3
TPR = 0 / (0 + 3) = 0
FPR = 0 / (0 + 4) = 0
Now, let's plot the TPR against FPR for each threshold:
(1, 1) (0.75, 1) (0.25, 0.67) (0.25, 0.33) (0, 0.33) (0, 0) (0, 0)
The points are not sorted by FPR, so we should sort them by FPR:
(0, 0) (0, 0) (0, 0.33) (0.25, 0.33) (0.25, 0.67) (0.75, 1) (1, 1)
"""
import numpy as np
from sklearn.metrics import roc_curve, auc

# Given data points: (x8, y)
data_points = [(-0.4, 0), (-0.1, 0), (0.3, 1), (0.6, 1), (0.7, 0), (0.8, 1), (0.9, 0)]

# Separate the data into predictions (x8) and true labels (y)
predictions, true_labels = zip(*data_points)

# Calculate the FPR, TPR, and thresholds using sklearn's roc_curve function
fpr, tpr, thresholds = roc_curve(true_labels, predictions)

# Calculate the AUC
roc_auc = auc(fpr, tpr)

# Print the FPR, TPR, and AUC
print('16.3')
print("FPR:", fpr)
print("TPR:", tpr)
print("AUC:", roc_auc)
print('\n')

# 3
"""
Given the number of true negatives (TN) at θ = 0.3 is 489, the TPR at this threshold is 0.412, and the FPR at this threshold is 0.164, we can find the approximate number of true positives (TP) using the following steps:

1- Calculate the total number of negatives (N) and positives (P) in the dataset.

2- Use the FPR and TPR values to calculate the number of false positives (FP) and true positives (TP).

Step 1: Calculate N and P:
TN + FP + TP + FN = 1000 (total number of observations)
Since FPR = FP / (FP + TN) and TPR = TP / (TP + FN), we can rewrite the equation as:
TN + (FPR * TN) + TP + (TPR * TP) = 1000

Step 2: Calculate FP and TP:
Given TN = 489 and FPR = 0.164, we can calculate FP as follows:
FP = FPR * (FP + TN) => FP = 0.164 * (FP + 489)

Given TPR = 0.412, we can calculate TP as follows:
TP = TPR * (TP + FN) => TP = 0.412 * (TP + (1000 - (TP + TN + FP)))

Now, we have two equations:
FP = 0.164 * (FP + 489)
TP = 0.412 * (TP + (1000 - (TP + TN + FP)))

We can solve this system of equations to find the approximate value of TP:
"""
from sympy import symbols, Eq, solve

FP, TP = symbols("FP TP")
TN = 489
FPR = 0.164
TPR = 0.412
N = 1000

eq1 = Eq(FP, FPR * (FP + TN))
eq2 = Eq(TP, TPR * (TP + (N - (TP + TN + FP))))

solution = solve((eq1, eq2), (FP, TP))

print("16.3")
# Print the equations
print("Equation 1:", eq1)
print("Equation 2:", eq2)
# Print TP
print("Approximate number of true positives (TP):", solution[TP])
print("\n")

######################################
# 17
######################################
print("Section 17:")
# 1
"""
In this problem, we are asked to find the weights of the observations after one round of AdaBoost, given that the training sets are sampled without replacement.

First, let's find the initial weights, w_0:

Since there are 4 observations, the initial weights for each observation will be 1/N = 1/4 = 0.25.

Next, let's compute the error for the classifier:

Error rate (e) is the weighted sum of incorrect predictions:

e = (w[1] * 0 + w[2] * 1 + w[3] * 0 + w[4] * 0) = 0.25 * 1 = 0.25

Next, let's find the classifier's coefficient (alpha):

alpha = 0.5 * ln((1 - e) / e) = 0.5 * ln((1 - 0.25) / 0.25) = 0.5493

Now, let's update the weights:

For correct predictions, the new weight w_new = w_old * exp(-alpha)
For incorrect predictions, the new weight w_new = w_old * exp(alpha)

So the new weights are:
w[1] = 0.25 * exp(-0.5493) = 0.1667
w[2] = 0.25 * exp(0.5493) = 0.5
w[3] = 0.25 * exp(-0.5493) = 0.1667
w[4] = 0.25 * exp(-0.5493) = 0.1667

The resulting weights are approximately:
w = [0.167, 0.5, 0.167, 0.167]

Thus, the correct answer is option B.
"""
import numpy as np

# Calculate the error ε1
misclassified = 1
total_observations = 4
epsilon1 = misclassified / total_observations

# Compute α1
alpha1 = 0.5 * np.log((1 - epsilon1) / epsilon1)

# Calculate the un-normalized weights
weights_unnormalized = [np.exp(-alpha1), np.exp(alpha1), np.exp(-alpha1), np.exp(-alpha1)]

# Normalize the weights
weights_normalized = [weight / sum(weights_unnormalized) for weight in weights_unnormalized]
print('17.1')
print(weights_normalized)
print('\n')

# 2
"""
The correct statement is B: For each round of bagging, it is expected that only a subset of the observations are used for training.

Bagging (Bootstrap Aggregating) is an ensemble method that creates multiple models by training them on different 
subsets of the training data. These subsets are created by random sampling with replacement. In each round of 
bagging, a subset is selected, and a model is trained on this subset. The final prediction is obtained by combining 
the predictions of all the models (e.g., by averaging or voting). 

The other options are incorrect because:

A: In boosting, misclassified observations are given more importance in the next round, not less.

C: Boosting does not use leave-one-out cross-validation to learn which observations to sample in the next round. Instead, it adjusts the weights of the samples based on the errors made by the current model.

D: When combining multiple classifiers using bagging, the final prediction is made by aggregating the predictions of all classifiers, not by selecting the classifier with the best performance.
"""

######################################
# 18
######################################
print("Section 18:")
# 1
"""
To determine which dendrogram corresponds to the clustering, we need to perform hierarchical clustering using single linkage on the given pairwise distances. Single linkage calculates the distance between two clusters as the minimum distance between any two points in the clusters.

Here is a step-by-step explanation for the hierarchical clustering process:
    Start with 8 clusters, each containing one observation.
    Find the pair of clusters with the smallest distance between them, which is the smallest value in the pairwise distance matrix.
    Merge the pair of clusters with the smallest distance into a single cluster.
    Update the pairwise distance matrix to reflect the merging of the clusters.
    Repeat steps 2-4 until all observations are in a single cluster.
    
There are several ways to measure the distance between clusters, which are called linkage methods:
    Single linkage (minimum linkage): The distance between two clusters is defined as the minimum distance between any two points in the clusters.
    Complete linkage (maximum linkage): The distance between two clusters is defined as the maximum distance between any two points in the clusters.
    Average linkage: The distance between two clusters is defined as the average distance between all pairs of points in the clusters.
    Ward's method: The distance between two clusters is defined as the increase in the sum of squared distances within the clusters after merging them.
"""
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

# Pairwise distances from Table 18.1
dist_matrix = np.array([
    [0, 2.39, 1.73, 0.96, 3.46, 4.07, 4.27, 5.11],
    [2.39, 0, 1.15, 1.76, 2.66, 5.36, 3.54, 4.79],
    [1.73, 1.15, 0, 1.52, 3.01, 4.66, 3.77, 4.90],
    [0.96, 1.76, 1.52, 0, 2.84, 4.25, 3.80, 4.74],
    [3.46, 2.66, 3.01, 2.84, 0, 4.88, 1.41, 2.96],
    [4.07, 5.36, 4.66, 4.25, 4.88, 0, 5.47, 5.16],
    [4.27, 3.54, 3.77, 3.80, 1.41, 5.47, 0, 2.88],
    [5.11, 4.79, 4.90, 4.74, 2.96, 5.16, 2.88, 0]
])

# Convert the distance matrix to a condensed form
condensed_dist = dist_matrix[np.triu_indices(8, k=1)]

# Perform hierarchical clustering using single linkage
Z = linkage(condensed_dist, method='single')

# Plot the dendrogram
plt.figure()
dendrogram(Z, labels=["O1", "O2", "O3", "O4", "O5", "O6", "O7", "O8"])
plt.title("Dendrogram-section-18-01")
plt.show()

# 2
"""
To solve this question using pen and paper, we need to apply the K-means clustering algorithm with K = 3 and initial cluster centers μ1 = 4, μ2 = 7, and μ3 = 14. The K-means algorithm proceeds as follows:

Assign each observation to the nearest cluster center.
Update the cluster centers by calculating the mean of the observations in each cluster.
Repeat steps 1 and 2 until the cluster assignments do not change or a maximum number of iterations is reached.
Let's go through the steps using the given data set X = {3, 6, 7, 9, 10, 11, 14}.

Initial cluster centers:
μ1 = 4, μ2 = 7, μ3 = 14

Step 1: Assign observations to the nearest cluster center:

Cluster 1 (μ1 = 4): 3
Cluster 2 (μ2 = 7): 6, 7, 9, 10
Cluster 3 (μ3 = 14):  11, 14
Step 2: Update cluster centers by calculating the mean of the observations in each cluster:

μ1 = mean(3) = 3.00
μ2 = mean(6, 7, 9, 10) = 8 
μ3 = mean(11, 14) = 12.5 
Step 1 (repeat): Assign observations to the nearest updated cluster center:

Cluster 1 (μ1 = 3.00): 3
Cluster 2 (μ2 = 8): 6, 7, 9, 10 
Cluster 3 (μ3 = 12.5): 11, 14
Step 2 (repeat): Update cluster centers again:

μ1 = mean(3) = 3.00
μ2 = mean(6, 7, 9, 10) = 8 
μ3 = mean(11, 14) = 12.5 
Since the cluster assignments did not change after the second iteration, the K-means algorithm has converged, and 
the final cluster centers are μ1 = 3.00, μ2 = 8, and μ3 = 12.5. A-option.
"""
import numpy as np
from sklearn.cluster import KMeans

data = np.array([3, 6, 7, 9, 10, 11, 14]).reshape(-1, 1)
init_centers = np.array([4, 7, 14]).reshape(-1, 1)

kmeans = KMeans(n_clusters=3, init=init_centers, n_init=1, random_state=42)
kmeans.fit(data)

final_centers = kmeans.cluster_centers_.round(2)
print('2')
print(final_centers)
print('\n')

# 3
"""
To solve this question using pen and paper, we need to apply hierarchical clustering using group average linkage on the given pairwise cityblock distances. The group average linkage computes the average distance between all pairs of points in the two clusters.

Here is the step-by-step procedure for group average linkage:

    Start with each observation as a separate cluster.
    Compute the average linkage distances between all pairs of clusters.
    Merge the two clusters with the smallest average linkage distance.
    Repeat steps 2 and 3 until only one cluster remains.
"""
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

distances = np.array([
    [0, 4, 7, 9, 5, 5, 5, 6],
    [4, 0, 7, 7, 7, 3, 7, 8],
    [7, 7, 0, 10, 6, 6, 4, 9],
    [9, 7, 10, 0, 8, 6, 10, 9],
    [5, 7, 6, 8, 0, 8, 6, 7],
    [5, 3, 6, 6, 8, 0, 8, 11],
    [5, 7, 4, 10, 6, 8, 0, 7],
    [6, 8, 9, 9, 7, 11, 7, 0]
])

# Convert the distance matrix to a condensed distance matrix (upper triangular)
condensed_distances = []
for i in range(len(distances)):
    for j in range(i + 1, len(distances)):
        condensed_distances.append(distances[i, j])

condensed_distances = np.array(condensed_distances)

# Perform hierarchical clustering using group average linkage
Z = linkage(condensed_distances, method='average')

# Plot the dendrogram
plt.figure(figsize=(8, 4))
dendrogram(Z, labels=['O1', 'O2', 'O3', 'O4', 'O5', 'O6', 'O7', 'O8'])
plt.xlabel('Observations 18.3')
plt.ylabel('Cityblock Distance')
plt.show()

"""
Sure, let's simplify the example even further by considering just 3 observations (A, B, and C) with the following pairwise distances:

A B C
A 0 2 5
B 2 0 3
C 5 3 0

Now, let's calculate the average linkage distances for all pairs of clusters:

Cluster X = {A}, Cluster Y = {B}:
Pair: A and B
Distance: d(A, B) = 2
Average linkage distance: 2 / 1 = 2
Cluster X = {A}, Cluster Y = {C}:
Pair: A and C
Distance: d(A, C) = 5
Average linkage distance: 5 / 1 = 5
Cluster X = {B}, Cluster Y = {C}:
Pair: B and C
Distance: d(B, C) = 3
Average linkage distance: 3 / 1 = 3
Now let's consider merging two clusters and then calculating the average linkage distance with the remaining cluster:

Suppose we merge Cluster {A} and {B} into a new cluster {A, B}. The new pairwise distances would look like this:

A B C
A 0 -1 5
B -1 0 3
C 5 3 0

(Note: The -1 indicates that the distance is not relevant since we're dealing with a merged cluster.)

Now, calculate the average linkage distance between the new cluster {A, B} and the remaining cluster {C}:

Pairs: (A, C) and (B, C)
Distances: d(A, C) = 5, d(B, C) = 3
Sum of distances: 5 + 3 = 8
Average linkage distance: 8 / 2 = 4
So, the average linkage distance between the merged cluster {A, B} and the remaining cluster {C} is 4.
"""
# 4
"""
In this problem, we have a clustering of 8 observations into 4 clusters. The goal is to calculate the purity of the clustering based on the class each observation belongs to (whether the person has a liver disease or not).

Here's the given clustering:

Cluster 1: A4, B1, B2, B3
Cluster 2: A1, B4
Cluster 3: A2
Cluster 4: A3
To calculate the purity, we first find the probability that a member of each cluster belongs to a particular class and then calculate the purity of each cluster (pi) by taking the maximum probability. Finally, we calculate the overall purity using the formula:

purity = sum((mi/m) * pi) for all clusters i

where mi is the number of observations in cluster i and m is the total number of observations.

Now let's calculate the purity for each cluster:

Cluster 1:
1 observation without liver disease (A4)
3 observations with liver disease (B1, B2, B3)
Probability of no liver disease (class A) in cluster 1: 1/4
Probability of liver disease (class B) in cluster 1: 3/4
Purity of cluster 1 (p1): max(1/4, 3/4) = 3/4
Cluster 2:
1 observation without liver disease (A1)
1 observation with liver disease (B4)
Probability of no liver disease (class A) in cluster 2: 1/2
Probability of liver disease (class B) in cluster 2: 1/2
Purity of cluster 2 (p2): max(1/2, 1/2) = 1/2
Cluster 3:
1 observation without liver disease (A2)
0 observations with liver disease
Probability of no liver disease (class A) in cluster 3: 1/1 = 1
Probability of liver disease (class B) in cluster 3: 0/1 = 0
Purity of cluster 3 (p3): max(1, 0) = 1
Cluster 4:
1 observation without liver disease (A3)
0 observations with liver disease
Probability of no liver disease (class A) in cluster 4: 1/1 = 1
Probability of liver disease (class B) in cluster 4: 0/1 = 0
Purity of cluster 4 (p4): max(1, 0) = 1
Now, let's calculate the overall purity:

purity = (4/8 * p1) + (2/8 * p2) + (1/8 * p3) + (1/8 * p4)
= (4/8 * 3/4) + (2/8 * 1/2) + (1/8 * 1) + (1/8 * 1)
= 3/8 + 1/8 + 1/8 + 1/8
= 6/8
= 3/4

So, the correct answer is:

D purity = 3/4
"""

# Given clustering
clusters = [
    ['A4', 'B1', 'B2', 'B3'],
    ['A1', 'B4'],
    ['A2'],
    ['A3']
]


# Function to calculate purity for each cluster
def cluster_purity(cluster):
    class_A_count = sum([1 for obs in cluster if obs.startswith('A')])
    class_B_count = sum([1 for obs in cluster if obs.startswith('B')])

    return max(class_A_count, class_B_count) / len(cluster)


# Calculate overall purity
total_observations = sum([len(cluster) for cluster in clusters])
purity = sum([(len(cluster) / total_observations) * cluster_purity(cluster) for cluster in clusters])

print("18.4")
print("Purity:", purity)
print("\n")

######################################
# 19
######################################
print("Section 19:")
# 2
"""
A. k-means and Gaussian Mixture Models are guaranteed to find the same solutions regardless of initialization.

This statement is incorrect. k-means and Gaussian Mixture Models can find different solutions due to different initialization and optimization methods. They may produce similar results, but there is no guarantee that they will always find the same solution.
B. The level at which clusters merge in the dendrogram in hierarchical clustering using minimum/single-, maximum/complete- or group average linkage can be determined by the proximities between all the observations.

This statement is correct. In hierarchical clustering, the distance between clusters (single-linkage, complete-linkage, or group average) determines the level at which they merge in the dendrogram.
C. In k-means the cluster centers are updated as the average of the observations belonging to the cluster regardless of the distance measure used.

This statement is incorrect. While cluster centers are updated as the average of the observations belonging to the cluster in k-means, the distance measure used can influence the clustering results.
D. A Gaussian Mixture Model with diagonal covariance matrix has the same number of free parameters as k-means.

This statement is incorrect. A Gaussian Mixture Model with diagonal covariance matrix has more free parameters than k-means because it includes both the means and the variances of each cluster.
The correct answer is B.
"""

# 3
"""
A. w1 = 0.5, σ2_1 = 2σ2_2, δ2_1 = 2δ2_2
This option suggests equal weights (w1 = 0.5) for both clusters. However, this contradicts the observation that the first cluster has higher density than the second cluster, indicating that the weights should be different.

B. w1 = 0.7, δ2_1 > σ2_2
This option suggests that the first cluster has a higher weight (w1 = 0.7), which is consistent with the observation of different densities in the scatter plot. However, it only provides information about the variance in the y-direction (δ2_i) and does not give any information about the variance in the x-direction (σ2_i). Due to this lack of information, it is difficult to conclude that option B is correct based solely on the provided information.

C. w1 = 0.7, σ2_1 = 20, δ2_2 = 1
This option suggests that the first cluster has a higher weight (w1 = 0.7), which is consistent with the observation of different densities in the scatter plot. Additionally, it provides specific values for the variances in both the x-direction (σ2_i) and y-direction (δ2_i) for both clusters. Given the scatter plot, the first cluster seems to have a larger variance in the x-direction, which matches this option.

D. w1 = 0.5, p(0, 0) < p(0, 7)
This option suggests equal weights (w1 = 0.5) for both clusters, which contradicts the observation of unequal densities in the scatter plot. The probability density comparison is true but not enough to make a conclusion.

To summarize, option C is the most likely correct answer because it is consistent with the observation of different densities in the scatter plot (indicating different weights) and provides specific values for the variances in both the x-direction and y-direction for both clusters.
"""

######################################
# 20
######################################
print("Section 20:")
# 1
import numpy as np

distances = np.array([5.11, 4.79, 4.90, 4.74, 2.96, 5.16, 2.88])
N = 7
M = 7

# Calculate the density at observation O8
density = (1 / N) * np.sum(np.exp(-0.5 * distances ** 2) / ((2 * np.pi) ** (M / 2)))

print("1")
print(f"p(O8) = {density:.10f}")
print("\n")

# 2
import numpy as np

distances = np.array([
    [0, 393.5, 68.1, 165.4, 271.8, 200.6, 210.9, 206.1, 166.3, 365.0],
    [393.5, 0, 411.3, 361.8, 478.6, 490.9, 409.2, 382.3, 391.1, 37.4],
    [68.1, 411.3, 0, 119.8, 208.4, 136.6, 152.8, 154.3, 111.1, 387.1],
    [165.4, 361.8, 119.8, 0, 137.5, 130.8, 62.1, 44.7, 32.5, 346.2],
    [271.8, 478.6, 208.4, 137.5, 0, 99.0, 76.8, 101.0, 116.4, 468.5],
    [200.6, 490.9, 136.6, 130.8, 99.0, 0, 100.1, 124.0, 100.5, 473.8],
    [210.9, 409.2, 152.8, 62.1, 76.8, 100.1, 0, 29.5, 45.2, 396.8],
    [206.1, 382.3, 154.3, 44.7, 101.0, 124.0, 29.5, 0, 44.6, 370.1],
    [166.3, 391.1, 111.1, 32.5, 116.4, 100.5, 45.2, 44.6, 0, 375.1],
    [365.0, 37.4, 387.1, 346.2, 468.5, 473.8, 396.8, 370.1, 375.1, 0]
])

K = 2

# Step 1: Find the 2 nearest neighbors of O1
O1_neighbors_indices = np.argsort(distances[0])[1:K + 1]


# Step 2: Calculate the KNN density for O1 and its 2 nearest neighbors
def knn_density(index, K):
    nearest_indices = np.argsort(distances[index])[1:K + 1]  # Exclude itself (start from 1 instead of 0)
    return K / np.sum(distances[index][nearest_indices])


O1_density = knn_density(0, K)
neighbors_densities = np.array([knn_density(i, K) for i in O1_neighbors_indices])

# Step 3: Calculate the average relative density for O1
ard_O1 = O1_density / np.mean(neighbors_densities)

print("2")
print(f"Average Relative Density for O1 (K=2): {ard_O1:.2f}")
print("\n")

# 3
"""
The given data set has eight data objects, and we need to detect outliers based on the standardization of the data and modeling it with a standard univariate normal distribution. The standardization of data is performed using the formula:

zn = (xn - mean(x)) / std(x)

where zn is the standardized value of the attribute xn, mean(x) is the mean of the attribute values, and std(x) is the standard deviation of the attribute values. Here, mean(x) = 6, and std(x) = 28.

After standardization, we can model the data with a standard univariate normal distribution N(0, 1), where the mean is 0 and the standard deviation is 1. We define an outlier as a data object with a standardized attribute value |zn| > c, where c is a constant such that the probability that |zn| > c is equal to α. In this case, we choose α = 0.0027, which corresponds to c = 3.00.

To determine which data objects are outliers, we need to calculate the standardized values zn for each attribute and compare them with the threshold value c. If the absolute value of zn is greater than c, then the corresponding data object is an outlier.

Using the given formula, we can calculate the standardized values zn for each attribute as follows:

z1 = (-3 - 6) / 28 = -0.39
z2 = (-5 - 6) / 28 = -0.46
z3 = (-8 - 6) / 28 = -0.71
z4 = (0 - 6) / 28 = -0.21
z5 = (75 - 6) / 28 = 2.57
z6 = (-4 - 6) / 28 = -0.43
z7 = (-1 - 6) / 28 = -0.25
z8 = (-6 - 6) / 28 = -0.43

The data objects with standardized values |zn| > c are considered outliers
The correct answer is option A, there are no outliers in the data set.
"""
data = [-3, -5, -8, 0, 75, -4, -1, -6]
mean_x = 6
std_x = 28
c = 3.00

# Standardize the data
standardized_data = [(x - mean_x) / std_x for x in data]

# Find the outliers
outliers = [x for i, x in enumerate(data) if abs(standardized_data[i]) > c]

print("3")
print("Outliers:", outliers)
print("\n")

# 4
import itertools

data = [
    [1, 0, 1, 0, 1, 0],
    [1, 0, 1, 0, 0, 1],
    [0, 1, 0, 1, 1, 0],
    [0, 1, 1, 0, 1, 0],
    [0, 1, 1, 0, 1, 0],
    [0, 1, 1, 0, 1, 0],
    [0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 1],
    [0, 1, 1, 0, 1, 0],
    [0, 1, 1, 0, 1, 0]
]

n_subjects = len(data)
min_support = 0.52

# Calculate the support for single-item itemsets
single_item_support = [sum(row[i] for row in data) / n_subjects for i in range(6)]

# Find frequent single-item itemsets
frequent_single_itemsets = [i for i, support in enumerate(single_item_support) if support > min_support]

# Find frequent itemsets of size 2 or more
frequent_itemsets = frequent_single_itemsets.copy()

for size in range(2, len(frequent_single_itemsets) + 1):
    for itemset in itertools.combinations(frequent_single_itemsets, size):
        support = sum(all(data[i][item] for item in itemset) for i in range(n_subjects)) / n_subjects
        if support > min_support:
            frequent_itemsets.append(itemset)

print("4")
# Print the results
item_names = ["YAY", "YAN", "OAY", "OAN", "PAY", "PAN"]
for itemset in frequent_itemsets:
    if isinstance(itemset, int):
        print(f"{{{item_names[itemset]}}}")
    else:
        print("{" + ", ".join(item_names[item] for item in itemset) + "}")
print("\n")

# 5
"""
To calculate the average relative KNN density of observation O8 for K = 3, we need to find the set of 3 nearest neighbors of O8 and calculate the density and relative density of O8 based on those neighbors.

Using the distance matrix given in Table 20.2, we can find the 3 nearest neighbors of O8 by sorting the distances in ascending order and selecting the 3 smallest distances. The distances from O8 to its neighbors are:

distance(O8, O4) = 4.74
distance(O8, O5) = 2.96
distance(O8, O7) = 2.88

Therefore, the 3 nearest neighbors of O8 are O5, O7, and O4.

To calculate the density of O8 based on its neighbors, we need to calculate the sum of distances from O8 to its neighbors:

density(O8, 3) = (1/3) * (2.88 + 2.96 + 4.74)^(-1) = 0.2836

To calculate the relative density of O8 based on its neighbors, we need to calculate the density of each neighbor and divide the density of O8 by the average of the neighbor densities:

density(O5, 3) = (1/3) * (1.41 + 2.66 + 2.84)^(-1) = 0.4342
density(O7, 3) = (1/3) * (1.41 + 2.88 + 3.54)^(-1) = 0.3831
density(O4, 3) = (1/3) * (0.96 + 1.76 + 1.52)^(-1) = 0.7075

average density = (0.4342 + 0.3831 + 0.7075) / 3 = 0.5083

a.r.d.(O8, 3) = density(O8, 3) / average density = 0.2836 / 0.5083 = 0.56

Therefore, the average relative KNN density of observation O8 for K = 3 nearest neighbors is 0.56. The correct answer is C.
"""
import numpy as np

distance_matrix = np.array([
    [0, 2.39, 1.73, 0.96, 3.46, 4.07, 4.27, 5.11],
    [2.39, 0, 1.15, 1.76, 2.66, 5.36, 3.54, 4.79],
    [1.73, 1.15, 0, 1.52, 3.01, 4.66, 3.77, 4.90],
    [0.96, 1.76, 1.52, 0, 2.84, 4.25, 3.80, 4.74],
    [3.46, 2.66, 3.01, 2.84, 0, 4.88, 1.41, 2.96],
    [4.07, 5.36, 4.66, 4.25, 4.88, 0, 5.47, 5.16],
    [4.27, 3.54, 3.77, 3.80, 1.41, 5.47, 0, 2.88],
    [5.11, 4.79, 4.90, 4.74, 2.96, 5.16, 2.88, 0]
])


def density(x, K, distance_matrix):
    sorted_distances = sorted(enumerate(distance_matrix[x]), key=lambda d: d[1])
    nearest_neighbors = [d[0] for d in sorted_distances[1:K + 1]]
    total_distance = sum(distance_matrix[x][y] for y in nearest_neighbors)
    return 1 / (total_distance / K)


def ard(x, K, distance_matrix):
    sorted_distances = sorted(enumerate(distance_matrix[x]), key=lambda d: d[1])
    nearest_neighbors = [d[0] for d in sorted_distances[1:K + 1]]
    density_x = density(x, K, distance_matrix)
    density_sum = sum(density(y, K, distance_matrix) for y in nearest_neighbors)
    return density_x / (density_sum / K)


K = 3
O8_index = 7
ard_O8 = ard(O8_index, K, distance_matrix)
print('5')
print(f"Average Relative Density for O8 (K=3): {ard_O8:.2f}")
print("\n")

# 6
"""
To calculate the a.r.d. of observation o1 using K = 1 nearest neighbours, we first need to find the nearest neighbour for o1 and calculate the density of o1.

From table 20.6, we can see that the nearest neighbour for o1 is o2 with a cityblock distance of 4. So, the density of o1 is:

density(o1, K=1) = 1 / (1 * d(o1, o2)) = 1 / 4

Next, we need to find the density of o1's nearest neighbour (o2) with K = 1:

The nearest neighbour for o2 is o6 with a cityblock distance of 3. So, the density of o2 is:

density(o2, K=1) = 1 / (1 * d(o2, o6)) = 1 / 3

Now, we can calculate the a.r.d. of o1 using K = 1 nearest neighbours:

a.r.d(o1, K=1) = density(o1, K=1) / (1 * density(o2, K=1)) = (1/4) / (1/3)

To solve this fraction, we can multiply both the numerator and the denominator by 12 (the least common multiple of 3 and 4):

(1/4) / (1/3) = (3/12) / (4/12) = 3/4

So, the a.r.d. of o1 with K = 1 nearest neighbours is 3/4.

The correct answer is D a.r.d(x = o1, K = 1) = 3/4.


When K = 2, we need to find the two nearest neighbours for each observation and calculate their average distances to find the density. Here's the step-by-step theoretical solution for a.r.d(o1, K=2):

Find the two nearest neighbours for o1 and their distances:
From table 20.6, the two nearest neighbours for o1 are o2 (distance 4) and o6 (distance 5).

Calculate the density of o1 with K = 2:
density(o1, K=2) = 1 / [(1/2) * (d(o1, o2) + d(o1, o6))] = 1 / [(1/2) * (4 + 5)] = 1 / 4.5

Find the two nearest neighbours and their densities for each of o1's nearest neighbours (o2 and o6):
For o2: The two nearest neighbours are o1 (distance 4) and o6 (distance 3).
density(o2, K=2) = 1 / [(1/2) * (d(o2, o1) + d(o2, o6))] = 1 / [(1/2) * (4 + 3)] = 1 / 3.5

For o6: The two nearest neighbours are o1 (distance 5) and o2 (distance 3).
density(o6, K=2) = 1 / [(1/2) * (d(o6, o1) + d(o6, o2))] = 1 / [(1/2) * (5 + 3)] = 1 / 4

Calculate the average density of o1's two nearest neighbours (o2 and o6):
average_density_neighbours = (density(o2, K=2) + density(o6, K=2)) / 2 = (1/3.5 + 1/4) / 2

Calculate the a.r.d. of o1 with K = 2:
a.r.d(o1, K=2) = density(o1, K=2) / average_density_neighbours = (1/4.5) / ((1/3.5 + 1/4) / 2)

To find the exact value, you can simplify the expression:

a.r.d(o1, K=2) = (1/4.5) / ((1/3.5 + 1/4) / 2) = (1/4.5) * 2 / (1/3.5 + 1/4) = 2/4.5 * 1 / (7/28) = (2/4.5) * (28/7) = (2 * 28) / (4.5 * 7) = 56 / 31.5

Thus, a.r.d(o1, K=2) = 56/31.5.
"""


def cityblock_distance(matrix, i, j):
    return matrix[i][j]


def nearest_neighbour(matrix, i):
    min_distance = float('inf')
    nearest_index = -1
    for j in range(len(matrix)):
        if i != j and matrix[i][j] < min_distance:
            min_distance = matrix[i][j]
            nearest_index = j
    return nearest_index, min_distance


def density(matrix, i, K):
    if K == 1:
        _, distance = nearest_neighbour(matrix, i)
        return 1 / distance
    else:
        # Add functionality for K > 1 if needed.
        pass


def ard(matrix, i, K):
    density_i = density(matrix, i, K)
    nearest_index, _ = nearest_neighbour(matrix, i)
    density_nearest = density(matrix, nearest_index, K)
    return density_i / density_nearest


distance_matrix = [
    [0, 4, 7, 9, 5, 5, 5, 6],
    [4, 0, 7, 7, 7, 3, 7, 8],
    [7, 7, 0, 10, 6, 6, 4, 9],
    [9, 7, 10, 0, 8, 6, 10, 9],
    [5, 7, 6, 8, 0, 8, 6, 7],
    [5, 3, 6, 6, 8, 0, 8, 11],
    [5, 7, 4, 10, 6, 8, 0, 7],
    [6, 8, 9, 9, 7, 11, 7, 0]
]

K = 1
i = 0  # Index of o1 in the distance matrix
ard_o1 = ard(distance_matrix, i, K)
print('6')
print(f"a.r.d(x = o1, K = 1) = {ard_o1}")
print("\n")

# 7
"""
To find the probability density at observation o1, we need to use the given mixture distribution formula with K = 8 mixture components, λ = 4, and d as the cityblock distance metric:

p(x|θ) = (1 / (2λ)) * exp(-d(x, θ) / λ)

We are given that each mixture component is weighted equally in the full mixture distribution. Since there are 8 components, each will have a weight of 1/8.

The probability density at observation o1 will be the sum of the weighted densities calculated for each of the 8 observations (θ1 to θ8) using the given formula:

p(o1) = (1/8) * Σ [p(o1|θi)] for i in {1, 2, ..., 8}

Let's calculate the probability density at o1 step by step:

Calculate the cityblock distance, d(o1, θi), between o1 and each θi (i = 1 to 8) using table 20.6.
Calculate the density p(o1|θi) for each θi using the given formula.
Calculate the weighted sum of densities to get the probability density at o1.
"""
import math


def mixture_density(x, theta, distance_matrix, lambd):
    i = x - 1
    j = theta - 1
    distance = cityblock_distance(distance_matrix, i, j)
    return (1 / (2 * lambd)) * math.exp(-distance / lambd)


def probability_density_o1(distance_matrix, lambd):
    density_sum = 0
    for theta in range(1, 9):
        density_sum += mixture_density(1, theta, distance_matrix, lambd)
    return (1 / 8) * density_sum


lambd = 4
p_o1 = probability_density_o1(distance_matrix, lambd)
print('7')
print(f"Probability density at o1 ≈ {p_o1:.3f}")
print('\n')
