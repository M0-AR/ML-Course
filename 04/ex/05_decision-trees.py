# 9 may 2015
def gini_impurity(probabilities):
    """
    Calculate Gini impurity for a given list of class probabilities.

    :param probabilities: List of class probabilities.
    :return: Gini impurity value.
    """
    return 1 - sum([p ** 2 for p in probabilities])


def impurity_gain(total_impurity, impurities, counts, total_count):
    """
    Calculate the impurity gain.

    :param total_impurity: Impurity of the original set.
    :param impurities: List of impurities for the subsets.
    :param counts: List of subset sizes.
    :param total_count: Size of the original set.
    :return: Impurity gain value.
    """
    weighted_impurities = sum([(counts[i] / total_count) * impurities[i] for i in range(len(impurities))])
    return total_impurity - weighted_impurities


if __name__ == "__main__":
    # Given data
    females = [55, 44, 88]
    males = [75, 59, 74]
    total_students = [sum(females), sum(males)]

    # Calculate probabilities
    prob_females = [f / sum(females) for f in females]
    prob_males = [m / sum(males) for m in males]
    prob_all = [(f + m) / sum(total_students) for f, m in zip(females, males)]

    # Calculate Gini impurities
    impurity_females = gini_impurity(prob_females)
    impurity_males = gini_impurity(prob_males)
    impurity_all = gini_impurity(prob_all)

    # Calculate impurity gain
    gain = impurity_gain(impurity_all, [impurity_females, impurity_males], total_students, sum(total_students))

    print(f"Impurity Gain (∆) = {gain:.10f}")


# 18 dec 2015
def classification_error_impurity(probs):
    """
    Calculate the impurity using classification error measure.

    :param probs: List of class probabilities.
    :return: Classification error impurity.
    """
    return 1 - max(probs)


def purity_gain(parent_probs, left_probs, right_probs, n_left, n_right):
    """
    Calculate the purity gain for a decision tree split.

    :param parent_probs: List of class probabilities for the parent node.
    :param left_probs: List of class probabilities for the left child node.
    :param right_probs: List of class probabilities for the right child node.
    :param n_left: Number of samples in the left child node.
    :param n_right: Number of samples in the right child node.
    :return: Purity gain value.
    """
    parent_impurity = classification_error_impurity(parent_probs)
    left_impurity = classification_error_impurity(left_probs)
    right_impurity = classification_error_impurity(right_probs)

    total_samples = n_left + n_right
    weighted_impurity = (n_left / total_samples) * left_impurity + (n_right / total_samples) * right_impurity

    return parent_impurity - weighted_impurity


if __name__ == "__main__":
    # Given data
    # For the root node (Parent)
    parent_probs = [9 / 15, 6 / 15]  # 9/15 for CKD=1, 6/15 for CKD=0

    # For the PC=0 (Left child)
    n_left = 7
    left_probs = [5 / 7, 2 / 7]  # 5/7 for CKD=1, 2/7 for CKD=0

    # For the PC=1 (Right child)
    n_right = 8
    right_probs = [7 / 8, 1 / 8]  # 7/8 for CKD=1, 1/8 for CKD=0

    # Calculate purity gain
    gain = purity_gain(parent_probs, left_probs, right_probs, n_left, n_right)

    print(f"Purity Gain (∆) = {gain:.2f}")


# 19 dec 2015
def compute_purity(cluster_assignments, labels):
    """
    Compute the purity of clustering for given data and labels.

    Parameters:
    - cluster_assignments (list of int): The cluster assigned to each data point.
                                         Assumes clusters are assigned integers starting from 0.
    - labels (list of int): The label/classification for each data point.
                            Assumes binary classification with labels 0 and 1.

    Returns:
    - float: Purity value of the clustering

    Example:
    # >>> round(compute_purity([0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]), 2)
    0.6
    """

    total_samples = len(cluster_assignments)
    unique_clusters = set(cluster_assignments)

    purity_sum = 0

    for cluster in unique_clusters:
        cluster_indices = [i for i, x in enumerate(cluster_assignments) if x == cluster]
        cluster_labels = [labels[i] for i in cluster_indices]

        cluster_size = len(cluster_labels)
        dominant_class_proportion = max(cluster_labels.count(0) / cluster_size, cluster_labels.count(1) / cluster_size)

        purity_sum += (cluster_size / total_samples) * dominant_class_proportion

    return purity_sum


if __name__ == "__main__":
    # Sample data
    # RBC attribute for each observation: O1 to O15
    # 0 means RBC=0, 1 means RBC=1
    rbc_values = [0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0]

    # CKD attribute for each observation: O1 to O15
    # 1 means CKD=1, 0 means CKD=0
    ckd_values = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]

    clusters = rbc_values
    labels = ckd_values
    purity = compute_purity(clusters, labels)
    print(f"Purity: {purity:.2f}")

# 9 may 2016

def purity_gain(I0, splits):
    """
    Compute the purity gain for a set of potential splits using classification error.

    Args:
    - I0 (float): The impurity of the parent node.
    - splits (list of dicts): A list containing information for each potential split.
      Each dictionary in the list represents a split and has the keys:
        - 'n': total number of observations in the split.
        - 'values': a list of tuples where each tuple contains the number of observations of class 0
                    and class 1 for each condition in the split.

    Returns:
    - list of float: The purity gain for each split.
    """

    deltas = []

    for split in splits:
        n = split['n']
        delta = I0
        for (y0, y1) in split['values']:
            max_val = max(y0, y1) / (y0 + y1)
            delta -= ((y0 + y1) / n) * (1 - max_val)
        deltas.append(delta)

    return deltas


def classification_error(p0, p1):
    """Compute the classification error impurity measure.

    Parameters:
    - p0: proportion of class 0 in the node.
    - p1: proportion of class 1 in the node.

    Returns:
    - Impurity measure based on classification error.
    """
    return 1 - max(p0, p1)


# Given data
obs_y0 = 45 + 47 + 8 # Observations for unoccupied rooms (y = 0)
obs_y1 = 1 + 66 + 33 # Observations for occupied rooms (y = 1)
total_obs = obs_y0 + obs_y1

# Compute probabilities
p0 = obs_y0 / total_obs
p1 = obs_y1 / total_obs

# Impurity of the parent node
# Calculate impurity
I0 = classification_error(p0, p1)

# Potential splits data
splits = [
    {'n': 200, 'values': [(45, 1), (47, 66), (8, 33)]},
    {'n': 200, 'values': [(76, 20), (16, 47), (8, 33)]},
    {'n': 200, 'values': [(25, 0), (55, 23), (20, 77)]}
]

# Calculate purity gains
deltas = purity_gain(I0, splits)

# Output the purity gains
for i, delta in enumerate(deltas, 1):
    print(f"Purity gain for Split {i}: {round(delta, 3)}")

# Select the best split
best_split = deltas.index(max(deltas)) + 1
print(f"\nBest Split based on Hunts Algorithm: Split {best_split}")

# 4 dec 2016
def classification_error(proportions):
    """
    Calculate the classification error impurity measure.

    Parameters:
    - proportions (list of float): Proportions of each class in a node.

    Returns:
    - float: Impurity measure based on classification error.
    """
    return 1 - max(proportions)


def purity_gain(before_split_counts, *after_split_counts):
    """
    Compute the purity gain for a split in a decision tree.

    Parameters:
    - before_split_counts (list of int): Class counts before the split.
    - *after_split_counts (list of lists): Variable number of lists representing class counts for each child node after the split.

    Returns:
    - float: Purity gain of the split.
    """
    total_samples = sum(before_split_counts)
    parent_proportions = [count / total_samples for count in before_split_counts]
    parent_impurity = classification_error(parent_proportions)

    weighted_child_impurity = sum(
        (sum(child_counts) / total_samples) * classification_error(
            [count / sum(child_counts) for count in child_counts])
        for child_counts in after_split_counts
    )

    return parent_impurity - weighted_child_impurity


# Given data
before_split = [70, 70, 70]
below_zero = [24, 70, 0]
above_or_equal_zero = [46, 0, 70]

# Calculate purity gain
gain = purity_gain(before_split, below_zero, above_or_equal_zero)
print(f"Purity Gain: {gain:.4f}")

# 10 may 2017
before_split = [17, 15]  # Sum of the values across z=0 and z=1 for each x5 value.

# Split A: 3 gear vs. 4 or 5 gears
split_a_1 = [13, 2]
split_a_2 = [4, 13]
gain_a = purity_gain(before_split, split_a_1, split_a_2)

# Split B: 3 or 4 gears vs. 5 gears
split_b_1 = [15, 12]
split_b_2 = [2, 3]
gain_b = purity_gain(before_split, split_b_1, split_b_2)

# Split C: 3 gears vs. 4 gears vs. 5 gears
# (Using the original splits since there are 3 branches)
split_c_1 = [13, 2]
split_c_2 = [2, 10]
split_c_3 = [2, 3]
gain_c = purity_gain(before_split, split_c_1, split_c_2, split_c_3)

print(f"Purity Gain for Split A: {gain_a:.4f}")
print(f"Purity Gain for Split B: {gain_b:.4f}")
print(f"Purity Gain for Split C: {gain_c:.4f}")

# Determining the correct statement
if gain_b > gain_a:
    print("A. Split B provides a higher purity gain than split A.")
if gain_c > gain_a:
    print("B. Split C provides a higher purity gain than split A.")
if max(gain_a, gain_b, gain_c) == 9/32:
    print("C. The best obtainable purity gain is 9/32.")
if gain_b > gain_c:
    print("D. Split B provides a higher purity gain than split C.")

# or
def compute_impurity(num_class_1, num_class_2):
    """
    Computes the classification error impurity.

    Parameters:
    - num_class_1: Number of samples in class 1.
    - num_class_2: Number of samples in class 2.

    Returns:
    - Impurity value.
    """
    total = num_class_1 + num_class_2
    p_class_1 = num_class_1 / total
    p_class_2 = num_class_2 / total
    return 1 - max(p_class_1, p_class_2)


def compute_purity_gain(parent_impurity, *splits):
    """
    Computes the purity gain for given splits.

    Parameters:
    - parent_impurity: Impurity of the parent node.
    - *splits: Variable length argument for splits. Each split is a tuple of
               (num_class_1, num_class_2).

    Returns:
    - Purity gain value.
    """
    total_samples = sum([sum(split) for split in splits])
    weighted_child_impurity = sum([(sum(split) / total_samples) * compute_impurity(*split) for split in splits])
    return parent_impurity - weighted_child_impurity


# Data from the table
low_mpg = [13, 2, 2]
high_mpg = [2, 10, 3]

# Compute impurity for the entire dataset (before any split)
parent_impurity = compute_impurity(sum(low_mpg), sum(high_mpg))

# Compute purity gain for each split
split_a_gain = compute_purity_gain(parent_impurity, (low_mpg[0], high_mpg[0]), (sum(low_mpg[1:]), sum(high_mpg[1:])))
split_b_gain = compute_purity_gain(parent_impurity, (sum(low_mpg[:2]), sum(high_mpg[:2])), (low_mpg[2], high_mpg[2]))
split_c_gain = compute_purity_gain(parent_impurity, (low_mpg[0], high_mpg[0]), (low_mpg[1], high_mpg[1]),
                                   (low_mpg[2], high_mpg[2]))

# Print the results
print(f"Split A purity gain: {split_a_gain:.4f}")
print(f"Split B purity gain: {split_b_gain:.4f}")
print(f"Split C purity gain: {split_c_gain:.4f}")

# Compare the results to determine the correct answer
answers = []
if split_b_gain > split_a_gain:
    answers.append("A. Split B provides a higher purity gain than split A.")
if split_c_gain > split_a_gain:
    answers.append("B. Split C provides a higher purity gain than split A.")
if max(split_a_gain, split_b_gain, split_c_gain) == 9 / 32:
    answers.append("C. The best obtainable purity gain is 9/32.")
if split_b_gain > split_c_gain:
    answers.append("D. Split B provides a higher purity gain than split C.")

print("\n".join(answers))

# 6 dec 2017
def gini_impurity(*classes):
    """
    Computes the Gini impurity.

    Parameters:
    - *classes: Variable length argument representing counts for each class.

    Returns:
    - Gini impurity value.
    """
    total = sum(classes)
    p = [cls / total for cls in classes]
    return 1 - sum([pi ** 2 for pi in p])


def purity_gain(parent_impurity, *splits):
    """
    Computes the purity gain.

    Parameters:
    - parent_impurity: Impurity of the parent node.
    - *splits: Variable length argument for splits. Each split is a list of class counts.

    Returns:
    - Purity gain value.
    """
    total_samples = sum([sum(split) for split in splits])
    weighted_child_impurity = sum([(sum(split) / total_samples) * gini_impurity(*split) for split in splits])
    return parent_impurity - weighted_child_impurity


# Initial data
low, mid, high = 18, 18, 18

# After split data
short_players = [6, 9, 3]
medium_players = [4, 6, 10]
tall_players = [8, 3, 5]

# Compute Gini impurity for the entire dataset (before any split)
parent_impurity = gini_impurity(low, mid, high)

# Compute purity gain for the split
gain = purity_gain(parent_impurity, short_players, medium_players, tall_players)

print(f"Purity gain (∆) of the split: {gain:.4f}")

# 11 may 2018
def classification_error(*classes):
    """
    Computes the classification error impurity.

    Parameters:
    - *classes: Variable length argument representing counts for each class.

    Returns:
    - Classification error impurity value.
    """
    total = sum(classes)
    p = [cls / total for cls in classes]
    return 1 - max(p)


def purity_gain(parent_impurity, *splits):
    """
    Computes the purity gain.

    Parameters:
    - parent_impurity: Impurity of the parent node.
    - *splits: Variable length argument for splits. Each split is a list of class counts.

    Returns:
    - Purity gain value.
    """
    total_samples = sum([sum(split) for split in splits])
    weighted_child_impurity = sum([(sum(split) / total_samples) * classification_error(*split) for split in splits])
    return parent_impurity - weighted_child_impurity


# Initial data: safe and unsafe airline companies
safe, unsafe = 32, 24

# After split data
few_incidences = [23, 8]
many_incidences = [9, 16]

# Compute classification error for the entire dataset (before any split)
parent_impurity = classification_error(safe, unsafe)

# Compute purity gain for the split
gain = purity_gain(parent_impurity, few_incidences, many_incidences)

print(f"Purity gain (∆) of the split: {gain:.4f}")

# 9 dec 2018
def classification_error(*classes):
    """
    Computes the classification error impurity.

    Parameters:
    - *classes: Variable length argument representing counts for each class.

    Returns:
    - Classification error impurity value.
    """
    total = sum(classes)
    p = [cls / total for cls in classes]
    return 1 - max(p)

def purity_gain(parent, *splits):
    """
    Computes the purity gain for given parent and splits using classification error.

    Parameters:
    - parent: Tuple representing class counts in the parent node.
    - *splits: Variable length argument representing class counts in each split.

    Returns:
    - Purity gain value.
    """
    parent_error = classification_error(*parent)
    split_errors = sum([(sum(split) / sum(parent)) * classification_error(*split) for split in splits])
    return parent_error - split_errors

# Given data
parent = (108 + 58, 112 + 75, 56 + 116)
split1 = (108, 112, 56)
split2 = (58, 75, 116)

delta = purity_gain(parent, split1, split2)
print(f"Purity gain: {delta:.3f}")

# 8 may 2019
def classification_error(*classes):
    """
    Computes the classification error impurity.

    Parameters:
    - *classes: Variable length argument representing counts for each class.

    Returns:
    - Classification error impurity value.
    """
    total = sum(classes)
    p = [cls / total for cls in classes]
    return 1 - max(p)

def purity_gain(parent, *splits):
    """
    Computes the purity gain for given parent and splits using classification error.

    Parameters:
    - parent: Tuple representing class counts in the parent node.
    - *splits: Variable length argument representing class counts in each split.

    Returns:
    - Purity gain value.
    """
    parent_error = classification_error(*parent)
    split_errors = sum([(sum(split) / sum(parent)) * classification_error(*split) for split in splits])
    return parent_error - split_errors

# Given data
parent = (263, 359, 358)
split1 = (143, 137, 54)
split2 = (223, 251, 197)

delta_split1 = purity_gain(parent, split1, tuple(parent[i]-split1[i] for i in range(3)))
delta_split2 = purity_gain(parent, split2, tuple(parent[i]-split2[i] for i in range(3)))

print(f"Purity gain for split x4 ≤ 0.43: {delta_split1:.4f}")
print(f"Purity gain for split x4 ≤ 0.55: {delta_split2:.4f}")

# 18 may 2020
def gini_impurity(*classes):
    """
    Computes the Gini impurity.

    Parameters:
    - *classes: Variable length argument representing counts for each class.

    Returns:
    - Gini impurity value.
    """
    total = sum(classes)
    p = [cls / total for cls in classes]
    return 1 - sum([pi ** 2 for pi in p])

def purity_gain(parent, *splits):
    """
    Computes the purity gain for given parent and splits using Gini impurity.

    Parameters:
    - parent: Tuple representing class counts in the parent node.
    - *splits: Variable length argument representing class counts in each split.

    Returns:
    - Purity gain value.
    """
    parent_impurity = gini_impurity(*parent)
    split_impurity = sum([(sum(split) / sum(parent)) * gini_impurity(*split) for split in splits])
    return parent_impurity - split_impurity

# Given data
# Count the observations in each class for f1 = 1 and f1 = 0
# yb=0 when f1=1 (o1 to o8, 8 observations) and yb=1 when f1=1 (o9, 1 observation)
f1_1 = (6, 1)
# yb=0 when f1=0 (o6, o7) and yb=1 when f1=0 (o10, o11, 2 observations)
f1_0 = (2, 2)

# There are 8 observations in total for yb = 0 and 3 for yb = 1.
parent = (8, 3)

delta = purity_gain(parent, f1_1, f1_0)

print(f"Purity gain for split on f1: {delta:.3f}")

# 18 dec 2020
def classification_error(*classes):
    """
    Computes the classification error impurity.

    Parameters:
    - *classes: Variable length argument representing counts for each class.

    Returns:
    - Classification error impurity value.
    """
    total = sum(classes)
    p = [cls / total for cls in classes]
    return 1 - max(p)

def purity_gain(parent_impurity, *splits):
    """
    Computes the purity gain for given parent impurity and splits using classification error.

    Parameters:
    - parent_impurity: Classification error impurity value of the parent node.
    - *splits: Tuples where the first value is the proportion of that split,
               and the second is the classification error of that split.

    Returns:
    - Purity gain value.
    """
    return parent_impurity - sum(proportion * split_impurity for proportion, split_impurity in splits)

# Observations counts
parent = (146, 119, 68)
left = (146, 119)
right = (68, )

# Impurities
parent_impurity = classification_error(*parent)
left_impurity = classification_error(*left)
right_impurity = classification_error(*right)

# Purity gain
delta = purity_gain(parent_impurity,
                    (sum(left) / sum(parent), left_impurity),
                     (sum(right) / sum(parent), right_impurity))

print(f"Purity gain for the split: {delta:.3f}")



# 14 may 2021
def impurity(y_values):
    """
    Computes the impurity measure for regression problems using the given y-values.

    Parameters:
    - y_values: List of target values

    Returns:
    - Impurity value.
    """
    mean = sum(y_values) / len(y_values)
    return sum([(y - mean)**2 for y in y_values]) / len(y_values)


# Data
x1_values = [-1.1, -0.8, 0.08, 0.18, 0.34, 0.6, 1.42, 1.68]
yr_values = [12, 5, 10, 23, 6, 17, 14, 13]

# Split data based on x1 > 0.26
v0_indices = [i for i, x in enumerate(x1_values) if x > 0.13]
v1_indices = [i for i, x in enumerate(x1_values) if x <= 0.26 and i in v0_indices]
v2_indices = [i for i in v0_indices if i not in v1_indices]

v0_values = [yr_values[i] for i in v0_indices]
v1_values = [yr_values[i] for i in v1_indices]
v2_values = [yr_values[i] for i in v2_indices]

# Impurity calculations
I_v0 = impurity(v0_values)
I_v1 = impurity(v1_values) if v1_values else 0  # Handle case where v1_values might be empty
I_v2 = impurity(v2_values)

# Purity gain
N = len(v0_values)
delta = I_v0 - ((len(v1_values) / N * I_v1) + (len(v2_values) / N * I_v2))

print(f"Purity gain for the split x1 > 0.26: {delta:.3f}")


# 12 dec 2021
def gini_impurity(classes):
    """
    Calculate Gini impurity for a list of class labels.

    Parameters:
    - classes: List of class labels

    Returns:
    - Gini impurity.
    """
    total_samples = len(classes)
    if total_samples == 0:
        return 0
    probs = [classes.count(cls) / total_samples for cls in set(classes)]
    return 1 - sum([p**2 for p in probs])

# Dataset from Table 3
f2_values = [0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1]
class_labels = ["C1", "C1", "C2", "C2", "C2", "C2", "C2", "C3", "C3", "C3", "C3"]

# Calculate initial Gini impurity at root node
root_impurity = gini_impurity(class_labels)

# Split based on f2
left_classes = [class_labels[i] for i, value in enumerate(f2_values) if value == 0]
right_classes = [class_labels[i] for i, value in enumerate(f2_values) if value == 1]

# Calculate Gini impurity for splits
left_impurity = gini_impurity(left_classes)
right_impurity = gini_impurity(right_classes)

# Compute impurity gain
total_samples = len(f2_values)
delta = root_impurity - (len(left_classes) / total_samples * left_impurity + len(right_classes) / total_samples * right_impurity)

print(f"Impurity gain for the split on f2: {delta:.4f}")
