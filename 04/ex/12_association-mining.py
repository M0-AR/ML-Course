# 16 - dec 2016
def association_rule_confidence(joint_probability: float, antecedent_probability: float) -> float:
    """
    Calculate the confidence of an association rule.

    Parameters:
    - joint_probability (float): P(A, B) - Probability of both the antecedent and consequent happening together.
    - antecedent_probability (float): P(A) - Probability of the antecedent happening.

    Returns:
    - float: Confidence of the association rule, given as a percentage.

    Example:
    To compute the confidence of the rule {x1, x2} -> {y}:
    joint_prob = P(y=1, x1=1, x2=1)
    antecedent_prob = P(x1=1, x2=1)
    confidence = association_rule_confidence(joint_prob, antecedent_prob)
    print(confidence)  # Outputs the confidence percentage
    """

    # Handle edge cases where the antecedent_probability is zero to avoid division by zero
    if antecedent_probability == 0:
        return 0.0

    # Calculate the confidence
    confidence = joint_probability / antecedent_probability

    # Return confidence as a percentage
    return confidence * 100.0


# Test
joint_prob = 1 / 14
antecedent_prob = 1 / 14
confidence = association_rule_confidence(joint_prob, antecedent_prob)
print(f"Confidence: {confidence:.1f}%")

# 18 may 2017
# Test
joint_prob = 2 / 8
antecedent_prob = 3 / 8
confidence = association_rule_confidence(joint_prob, antecedent_prob)
print(f"Confidence: {confidence:.1f}%")
