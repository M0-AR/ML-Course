# 23 dec 2020
from scipy.stats import binom


def mcnemars_test(n1, n2):
    """
    Perform McNemar's test to calculate the p-value for two classifiers.

    Args:
    - n1 (int): Number of times Model 1 is correct and Model 2 is incorrect.
    - n2 (int): Number of times Model 1 is incorrect and Model 2 is correct.

    Returns:
    - p_value (float): Computed p-value for McNemar's test.
    """
    # Compute the value for the binomial CDF
    cdf_value = binom.cdf(min(n1, n2), n1 + n2, 0.5)

    # Calculate the p-value
    p_value = 2 * cdf_value

    return p_value


# Data from the table
n1 = 28  # Total number of times that M1 is correct and M2 is incorrect
n2 = 35  # Total number of times that M1 is incorrect and M2 is correct

# Compute the p-value
p_value = mcnemars_test(n1, n2)
print(f"The p-value from McNemar's test is approximately: {p_value:.2f}")

# -------
# 2023
def compute_theta(n1, n2):
    """
    Compute theta based on the formula.

    Parameters:
    - n1: Number of times Model A is correct and Model B/C is wrong.
    - n2: Number of times Model A is wrong and Model B/C is correct.

    Returns:
    - Theta value.
    """
    return (n1 - n2) / (n1 + n2)


# For MA vs MB
n1_AB = 38
n2_AB = 42

# For MA vs MC
n1_AC = 42
n2_AC = 38

theta_AB = compute_theta(n1_AB, n2_AB)
theta_AC = compute_theta(n1_AC, n2_AC)

# Checking conclusions
p_values_equal = n1_AB == n2_AC and n2_AB == n1_AC
theta_values_opposite = theta_AB == -theta_AC

if p_values_equal and theta_values_opposite:
    print("Conclusion: p_AB = p_AC and Θ_AB = -Θ_AC")
else:
    print("Conclusion does not match the expected result.")

print(f"Theta_AB: {theta_AB:.4f}")
print(f"Theta_AC: {theta_AC:.4f}")

from scipy.stats import chi2

def mcnemar_p_value(n1, n2):
    """
    Compute p-value using McNemar's test.

    Parameters:
    - n1: Number of times Model A is correct and Model B/C is wrong.
    - n2: Number of times Model A is wrong and Model B/C is correct.

    Returns:
    - p-value.
    """
    Q = (n1 - n2) ** 2 / (n1 + n2)
    # Use chi2 distribution to compute the p-value with 1 degree of freedom.
    # Since chi2.sf returns the survival function, it's equal to 1-cdf.
    p_value = chi2.sf(Q, 1)
    return p_value


# For MA vs MB
n1_AB = 38
n2_AB = 42
p_AB = mcnemar_p_value(n1_AB, n2_AB)

# For MA vs MC
n1_AC = 42
n2_AC = 38
p_AC = mcnemar_p_value(n1_AC, n2_AC)

print(f"p_AB: {p_AB:.4f}")
print(f"p_AC: {p_AC:.4f}")
print(f"Theta_AB: {theta_AB:.4f}")
print(f"Theta_AC: {theta_AC:.4f}")