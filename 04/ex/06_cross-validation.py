# --------------- CV -----------------
"""
Total time for the entire procedure: 31956000.00 units of time.
Total number of models trained: 205
Total number of models trained: 148712
The best strategy within the computational budget is: K1 = 14, K2 = 2
The procedure that satisfies the constraint is: K1 = 2, K2 = 10
Total time required for the 2-level cross-validation procedure is: 1020 minutes.
Total number of models trained: 208
"""


# 27 - fall 2014
def compute_training_testing_time(Ntrain, Ntest):
    """
    Compute the time taken to train and test a model.

    Parameters:
    - Ntrain (int): Number of training samples.
    - Ntest (int): Number of testing samples.

    Returns:
    - float: Total time taken to train and test the model.
    """
    training_time = Ntrain ** 2
    testing_time = 0.5 * Ntest ** 2

    return training_time + testing_time


def total_procedure_time(N, validation_percentage, L, K):
    """
    Compute the total time taken for cross-validation and hold-out testing.

    Parameters:
    - N (int): Total number of samples.
    - validation_percentage (float): Percentage of data to be used for validation. Should be between 0 and 1.
    - L (int): Number of regularization strengths to consider.
    - K (int): Number of folds for cross-validation.

    Returns:
    - float: Total time taken for the entire procedure.
    """
    # Split dataset into cross-validation set and validation set
    Dcv = N * (1 - validation_percentage)
    Dval = N * validation_percentage

    # Size of datasets for each fold in cross-validation
    Dcv_train = Dcv * (1 - (1 / K))
    Dcv_test = Dcv * (1 / K)

    # Time for cross-validation
    Tcv = L * K * compute_training_testing_time(Dcv_train, Dcv_test)

    # Time for testing and training on the hold-out dataset
    Tval = compute_training_testing_time(Dcv, Dval)

    # Total time
    return Tcv + Tval


if __name__ == "__main__":
    N = 1000
    validation_percentage = 0.2
    L = 6
    K = 10

    total_time = total_procedure_time(N, validation_percentage, L, K)
    print(f"Total time for the entire procedure: {total_time:.2f} units of time.")


# 27 - spring 2015
def number_of_models_trained(K1, K2, L):
    """
    Compute the total number of models trained for a two-level cross-validation strategy.

    Parameters:
    - K1 (int): Number of outer folds.
    - K2 (int): Number of inner folds.
    - L (int): Number of different settings (e.g., number of hidden units).

    Returns:
    - int: Total number of models trained.
    """
    # For each outer fold, K2 models are trained for each setting,
    # plus one more for estimating the generalization error.
    return K1 * (K2 * L + 1)


if __name__ == "__main__":
    K1 = 5
    K2 = 10
    L = 4  # Number of hidden unit settings

    total_models_trained = number_of_models_trained(K1, K2, L)
    print(f"Total number of models trained: {total_models_trained}")


# 20 - fall 2015
def number_of_models_trained(N, K_inner, L):
    """
    Compute the total number of models trained for a two-level cross-validation strategy.

    Parameters:
    - N (int): Total number of observations.
    - K_inner (int): Number of inner folds.
    - L (int): Number of different settings (e.g., different feature combinations).

    Returns:
    - int: Total number of models trained.
    """
    # For leave-one-out outer cross-validation, there will be N splits.
    # For each outer split:
    # 1. K_inner models are trained for each setting in the inner loop.
    # 2. One additional model is trained based on the optimal feature combination from the inner loop.
    models_per_outer_fold = K_inner * L + 1

    return N * models_per_outer_fold


if __name__ == "__main__":
    N = 232  # Total number of observations
    K_inner = 5  # Five-fold inner cross-validation
    L = 2 ** 7  # 128 different feature combinations

    total_models_trained = number_of_models_trained(N, K_inner, L)
    print(f"Total number of models trained: {total_models_trained}")


# 27 - spring 2016
def models_trained(K1, K2, L):
    """
    Compute the total number of models trained for a two-level cross-validation strategy.

    Parameters:
    - K1 (int): Number of outer folds.
    - K2 (int): Number of inner folds.
    - L (int): Number of different settings (e.g., pruning levels).

    Returns:
    - int: Total number of models trained.
    """
    # For each K1 outer fold:
    # 1. K2 models are trained for each setting in the inner loop.
    # 2. One additional model is trained based on the optimal setting from the inner loop.
    return K1 * (K2 * L + 1)


def most_models_within_budget(options, L, budget):
    """
    Determine the option that trains the most models within a given budget.

    Parameters:
    - options (list of tuples): List of (K1, K2) pairs representing the cross-validation strategies.
    - L (int): Number of different settings.
    - budget (int): Maximum number of models that can be trained.

    Returns:
    - tuple: (K1, K2) pair that trains the most models within the budget.
    """
    max_models = 0
    best_option = None

    for (K1, K2) in options:
        total_models = models_trained(K1, K2, L)
        if total_models <= budget and total_models > max_models:
            max_models = total_models
            best_option = (K1, K2)

    return best_option


if __name__ == "__main__":
    L = 3  # Number of pruning levels
    budget = 100
    options = [(6, 5), (3, 11), (14, 2), (4, 9)]

    best_strategy = most_models_within_budget(options, L, budget)
    print(f"The best strategy within the computational budget is: K1 = {best_strategy[0]}, K2 = {best_strategy[1]}")


# 24 - fall 2016
def models_trained(K1, K2, S):
    """
    Compute the total number of models trained for a two-level cross-validation strategy.

    Parameters:
    - K1 (int): Number of outer folds.
    - K2 (int): Number of inner folds.
    - S (int): Number of different settings (e.g., parameter settings).

    Returns:
    - int: Total number of models trained.
    """
    # For each K1 outer fold:
    # 1. K2 models are trained for each setting in the inner loop.
    # 2. One additional model is trained based on the optimal setting from the inner loop.
    return K1 * (K2 * S + 1)


def procedure_within_limit(options, S, limit):
    """
    Determine the procedures that train the least number of models and is within a given limit.

    Parameters:
    - options (list of tuples): List of (K1, K2) pairs representing the cross-validation strategies.
    - S (int): Number of different settings.
    - limit (int): Maximum number of models that can be trained.

    Returns:
    - tuple: (K1, K2) pair that trains the least number of models within the limit.
    """
    viable_options = {}
    for (K1, K2) in options:
        total_models = models_trained(K1, K2, S)
        if total_models <= limit:
            viable_options[(K1, K2)] = total_models

    return min(viable_options, key=viable_options.get, default=None)


if __name__ == "__main__":
    S = 3  # Number of different parameter settings
    limit = 65
    options = [
        (5, 5),  # Option A
        (100_000_000, 1),  # Option B
        (10, 2),  # Option C
        (2, 10)  # Option D
    ]

    best_strategy = procedure_within_limit(options, S, limit)
    if best_strategy:
        print(f"The procedure that satisfies the constraint is: K1 = {best_strategy[0]}, K2 = {best_strategy[1]}")
    else:
        print("None of the options satisfy the constraint.")


# 24 - spring 2017

# 23 - fall 2018
def training_time(K1, K2, S, time_per_model):
    """
    Compute the total training time for a two-level cross-validation strategy.

    Parameters:
    - K1 (int): Number of outer folds.
    - K2 (int): Number of inner folds.
    - S (int): Number of different model settings (e.g., ways to encode documents).
    - time_per_model (int): Time required to train a single model.

    Returns:
    - int: Total training time in minutes.
    """
    # For each K1 outer fold:
    # 1. K2 models are trained for each setting in the inner loop.
    # 2. One additional model is trained based on the optimal setting from the inner loop.
    total_models = K1 * (K2 * S + 1)
    return total_models * time_per_model


if __name__ == "__main__":
    K1 = 3
    K2 = 4
    S = 4
    time_per_model = 20  # minutes

    total_time = training_time(K1, K2, S, time_per_model)
    print(f"Total time required for the 2-level cross-validation procedure is: {total_time} minutes.")


# 22 - fall 2019
def total_models_trained(K1, K2, S):
    """
    Compute the total number of models trained for a two-level cross-validation strategy.

    Parameters:
    - K1 (int): Number of outer folds.
    - K2 (int): Number of inner folds.
    - S (int): Number of different model settings.

    Returns:
    - int: Total number of models trained.
    """
    # For each K1 outer fold:
    # 1. K2 models are trained for each setting in the inner loop.
    # 2. One additional model is trained based on the optimal setting from the inner loop.
    return K1 * (K2 * S + 1)


if __name__ == "__main__":
    K1 = 4  # Number of outer folds.
    K2 = 5  # Number of inner folds.
    S = 5  # Number of settings for each model.

    # Total number of models for neural network.
    total_nn_models = total_models_trained(K1, K2, S)
    # Total number of models for logistic regression.
    total_lr_models = total_models_trained(K1, K2, S)
    # Summing up the models of both neural network and logistic regression.
    total_models = total_nn_models + total_lr_models

    print(f"Total number of models trained: {total_models}")


# 13 - may 2020
def total_time_required(K1, K2, S, train_time, test_time):
    """
    Compute the total time required for a 2-level cross-validation strategy.

    Parameters:
    - K1 (int): Number of outer folds.
    - K2 (int): Number of inner folds.
    - S (int): Number of different model settings.
    - train_time (int): Time taken to train a single model (in seconds).
    - test_time (int): Time taken to test a single model (in seconds).

    Returns:
    - int: Total time required (in seconds).
    """
    # Total number of models trained and tested
    total_models = K1 * (K2 * S + 1)

    # Total training and testing time
    total_time = total_models * (train_time + test_time)

    return total_time


if __name__ == "__main__":
    K1 = 4  # Number of outer folds.
    K2 = 7  # Number of inner folds.
    S = 3  # Number of model settings.
    train_time = 20  # Time to train a model (in seconds).
    test_time = 1  # Time to test a model (in seconds).

    total_time = total_time_required(K1, K2, S, train_time, test_time)

    print(f"Total time required: {total_time} seconds")


# 9 - may 2021
def backward_selection(features):
    """
    Simulates the backward selection process for feature selection.

    Args:
    - features (list): A list of features in the dataset.

    Returns:
    - int: Total number of models evaluated during the backward selection process.
    """
    total_models_evaluated = 0
    current_features = features.copy()

    # Initially, evaluate the model with all features
    total_models_evaluated += 1

    while len(current_features) > 1:
        # Evaluate a model for each feature removed
        total_models_evaluated += len(current_features)

        # Simulate the selection process by removing the last feature
        # (In a real-world scenario, you'd remove the feature that produces the best model when excluded)
        print()
        current_features.pop()

    return total_models_evaluated


# Example usage
features = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8"]
num_evaluated = backward_selection(features)
print(f"Number of models evaluated: {num_evaluated}")

# 23 - dec 2021

# ---------------------------------------------------
# ---------------------------------------------------
# ---------------------------------------------------
# 7 - may 2017

def number_of_trained_models(data_size: int, num_lambda_values: int, cv_type: str) -> int:
    """
    Computes the number of trained models based on data size, number of lambda values, and cross-validation type.

    Parameters:
    - data_size (int): Size of the dataset.
    - num_lambda_values (int): Number of different lambda values used for regularization.
    - cv_type (str): Type of cross-validation, can be 'leave-one-out' or 'k-fold', where k is the number of folds.

    Returns:
    - int: Total number of models trained.
    """

    if cv_type == "leave-one-out":
        num_models = data_size * num_lambda_values
    elif "-fold" in cv_type:
        k = int(cv_type.split("-")[0])  # Extract the number of folds
        num_models = (data_size // k) * num_lambda_values
    else:
        raise ValueError("Unsupported cross-validation type.")

    return num_models


if __name__ == "__main__":
    data_size = 32
    num_lambda_values = 9
    cv_type = "leave-one-out"

    total_models = number_of_trained_models(data_size, num_lambda_values, cv_type)
    print(f"Total number of models trained with {cv_type} cross-validation: {total_models}")

# 23 - dec 2017
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut, cross_val_score
import matplotlib.pyplot as plt


def ridge_regression_demo(X, y, lambdas):
    """
    Performs Ridge Regression for various values of λ and plots training and test error.

    Parameters:
    - X (np.array): The feature matrix.
    - y (np.array): The target vector.
    - lambdas (list): List of regularization parameters to evaluate.

    Returns:
    - dict: Weights of the model for each λ value.
    """

    training_errors = []
    test_errors = []
    weights = {}

    loo = LeaveOneOut()

    for lam in lambdas:
        model = Ridge(alpha=lam)

        # Training the model
        model.fit(X, y)

        training_error = ((y - model.predict(X)) ** 2).mean()
        test_error = -cross_val_score(model, X, y, cv=loo, scoring='neg_mean_squared_error').mean()

        training_errors.append(training_error)
        test_errors.append(test_error)

        weights[lam] = model.coef_

    # Plotting
    plt.plot(np.log10(lambdas), training_errors, 'o-', label='Training Error')
    plt.plot(np.log10(lambdas), test_errors, 'x-', label='Test Error')
    plt.xlabel('log10(λ)')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.title('Training and Test Error vs log10(λ)')
    plt.show()

    return weights


if __name__ == "__main__":
    # Example data
    np.random.seed(0)
    X = np.random.randn(54, 4)
    y = 2.76 - 0.37 * X[:, 0] + 0.01 * X[:, 1] + 7.67 * X[:, 2] + 7.67 * X[:, 3] + np.random.normal(0, 0.5, 54)

    lambdas = np.logspace(-3, 3, 20)
    weights = ridge_regression_demo(X, y, lambdas)
    for lam, w in weights.items():
        print(f"For λ = {lam}, weights are: {w}")

# 24 - dec 2017
import numpy as np
from itertools import combinations
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut, cross_val_score
import matplotlib.pyplot as plt


def ridge_regression(X, y, lambdas):
    """
    Performs Ridge Regression for various values of λ and returns test error.

    Parameters:
    - X (np.array): The feature matrix.
    - y (np.array): The target vector.
    - lambdas (list): List of regularization parameters to evaluate.

    Returns:
    - list: Test errors for each λ value.
    """

    test_errors = []

    loo = LeaveOneOut()

    for lam in lambdas:
        model = Ridge(alpha=lam)
        test_error = -cross_val_score(model, X, y, cv=loo, scoring='neg_mean_squared_error').mean()
        test_errors.append(test_error)

    return test_errors


def exhaustive_feature_evaluation(X, y):
    """
    Evaluates all feature combinations for a linear regression model.

    Parameters:
    - X (np.array): The feature matrix.
    - y (np.array): The target vector.

    Returns:
    - list: Test errors for each feature combination.
    """

    test_errors = []
    num_features = X.shape[1]

    loo = LeaveOneOut()

    for r in range(1, num_features + 1):
        for indices in combinations(range(num_features), r):
            X_subset = X[:, indices]
            model = Ridge(alpha=0)  # No regularization
            test_error = -cross_val_score(model, X_subset, y, cv=loo, scoring='neg_mean_squared_error').mean()
            test_errors.append(test_error)

    return test_errors


if __name__ == "__main__":
    # Example data
    np.random.seed(0)
    X = np.random.randn(54, 4)
    y = np.random.randn(54)

    lambdas = np.linspace(0, 5, 20)
    ridge_errors = ridge_regression(X, y, lambdas)
    exhaustive_errors = exhaustive_feature_evaluation(X, y)

    print(f"Number of models evaluated for ridge regression: {len(lambdas) * 54}")
    print(f"Number of models evaluated for exhaustive feature evaluation: {len(exhaustive_errors) * 54}")

# ---------------------------------------------------
# ---------------------------------------------------
# ---------------------------------------------------

# 9 - may 2018
def compute_max_hidden_units(max_models=1000,
                             inner_fold=10,
                             outer_fold=5,
                             initializations=3):
    """
    Calculate the maximum number of hidden units for a neural network training procedure
    using two-level cross-validation.

    Parameters:
    - max_models (int): Maximum number of models that can be trained given the computational budget.
    - inner_fold (int): Number of cross-validation folds in the inner loop.
    - outer_fold (int): Number of cross-validation folds in the outer loop.
    - initializations (int): Number of random initializations per model specification.

    Returns:
    - int: Maximum possible number of hidden units.
    """

    # Given the equation 3 * 5 * (10 * H + 1) <= max_models
    # Solve for H:
    H = (max_models / (initializations * outer_fold) - 1) / inner_fold

    # Return the floor value of H to ensure integer hidden units
    return int(H)

if __name__ == "__main__":
    max_hidden_units = compute_max_hidden_units()
    print(f"Maximum number of hidden units: {max_hidden_units}")



# 25 dec 2021
import math


def compute_max_folds(N, test_ratio, regularization_strengths, computational_budget):
    """
    Determine the maximum number of folds K for cross-validation given a computational budget.

    Parameters:
    - N (int): Total number of observations.
    - test_ratio (float): Proportion of the dataset to use for testing.
    - regularization_strengths (int): Number of regularization strengths considered.
    - computational_budget (int): Computational budget in time units.

    Returns:
    - int: Maximum number of cross-validation folds.
    """
    # Number of observations for outer loop training
    no = int(N * (1 - test_ratio))

    # Number of observations for outer loop testing
    mo = int(N * test_ratio)

    # Time used in the outer loop
    to = no * math.log2(no) + mo

    for K in range(1, N):
        # Number of observations for inner loop training
        ni = no * (K - 1) / K

        # Number of observations for inner loop testing
        mi = no / K

        # Ensure ni is positive
        if ni <= 0:
            continue

        # Time used in the inner loop
        ti = regularization_strengths * K * (ni * math.log2(ni) + mi)

        total_time = to + ti

        if total_time > computational_budget:
            return K - 1  # return the previous K value before exceeding the budget

    return 0  # if the loop runs for all K and doesn't find a solution


if __name__ == "__main__":
    N = 1000
    test_ratio = 0.2
    regularization_strengths = 3
    computational_budget = 200000

    max_k = compute_max_folds(N, test_ratio, regularization_strengths, computational_budget)
    print(f"Maximum number of cross-validation folds: {max_k}")
