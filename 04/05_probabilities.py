"""
02450ex_Fall2021_sol-1.pdf
Question 16
"""
"""This script first calculates the means of x1 and x2 for class C1. It then calculates the probability of class C1, 
and the conditional probabilities of x1 and x2 given class C1. These values are used to calculate the numerator of 
the Bayes formula. Finally, this numerator is divided by the provided denominator to get the final probability of 
class C1 given x1 and x2. The result is printed out as a percentage. """
from scipy.stats import norm

# Given values
x1_test = 32.0
x2_test = 14.0
sigma_square = 400 # variance
p_denominator = 0.00010141

# Class C1
x1_C1 = [38.0, 26.8]
x2_C1 = [15.1, 12.8]
num_samples_C1 = 2
total_samples = 11

# Calculate means
mu1_1 = sum(x1_C1) / len(x1_C1)
mu1_2 = sum(x2_C1) / len(x2_C1)

# Calculate class probability
p_C1 = num_samples_C1 / total_samples

# Calculate conditional probabilities
p_x1_given_C1 = norm.pdf(x1_test, mu1_1, sigma_square**0.5)
p_x2_given_C1 = norm.pdf(x2_test, mu1_2, sigma_square**0.5)

# Calculate numerator
numerator = p_x1_given_C1 * p_x2_given_C1 * p_C1

# Calculate final class probability
p_C1_given_x1_x2 = numerator / p_denominator

print(f"The probability that the oil comes from the region North Apulia (C1) is approximately {p_C1_given_x1_x2*100:.2f}%")