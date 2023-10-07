"""
By: Mohamad Ashmar

There are multiple sections of code, each independent of the others. To avoid any errors, you should run each section
separately, ensuring that you do not mix the code from different sections.

Some of the code below inspired from
https://github.com/jmontalvo94/02450_Introduction_to_ML/blob/master/3_scripts/project_2_jm.py
"""

from Tools.toolbox_02450.statistics import mcnemar, correlated_ttest

"""
Regression, part a: In this section, you are to solve a relevant regression problem
for your data and statistically evaluate the result. We will begin by examining the
most elementary model, namely linear regression.

1. Explain what variable is predicted based on which other variables and what
you hope to accomplish by the regression. Mention your feature transformation
choices such as one-of-K coding. Since we will use regularization momentarily,
apply a feature transformation to your data matrix X such that each column
has mean 0 and standard deviation 1.
"""
# import numpy as np
# import pandas as pd
#
# from Tools.toolbox_02450.statistics import correlated_ttest, mcnemar
#
# import numpy as np
# import pandas as pd
# import sklearn.linear_model as lm
# from sklearn.model_selection import train_test_split
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
#
# # Load csv file with data
# filename = 'saheart_1_withheader.csv'
# data = pd.read_csv(filename)
#
# # Extract 'adiposity' (target variable) and ['obesity', 'age'] (predictor variables)
# """
# In this regression analysis, we aim to predict 'adiposity' based on 'obesity' and 'age'.
# The 'adiposity' variable represents the amount of adipose tissue and is our dependent variable,
# whereas 'obesity' and 'age' are our independent variables used to predict 'adiposity'. The purpose
# of this regression is to understand how well 'obesity' and 'age' can predict 'adiposity' and to
# identify the relationship between these variables.
#
# We utilize feature scaling to ensure each predictor variable has a mean of 0 and a standard deviation of 1.
# This standardization process is crucial, especially when regularization will be applied in subsequent analyses
# since it ensures that each feature contributes equally to the regression model and avoids a feature with larger
# scale dominating the prediction.
# """
# y = data['adiposity'].values
# X = data[['obesity', 'age']].values
#
# # Apply feature scaling to the predictor variables (mean 0 and standard deviation 1)
# X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)
#
# # Splitting the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
#
# # Fit Linear Regression model
# """
# We implement a basic linear regression model with the training data (X_train and y_train).
# No polynomial or interaction terms are introduced (Km=1), maintaining a straightforward linear relationship
# between predictor variables and the dependent variable.
# """
# Km = 1  # No of terms for regression model
# model = lm.LinearRegression()
# model = model.fit(X_train, y_train)
#
# # Predict values
# y_est = model.predict(X_test)
#
# # Calculate R squared value
# """
# R squared (R^2) value represents the proportion of variance in the dependent variable that
# is predictable from the independent variables. It provides a measure of how well observed
# outcomes are replicated by the model, based on the proportion of total variation of outcomes
# explained by the model.
# """
# print('R squared value:', model.score(X_test, y_test))
#
# # Weights of the regression model
# """
# The intercept and coefficients derived from the regression model reveal how the dependent variable
# 'adiposity' changes with a one-unit change in 'obesity' and 'age', respectively, while holding the other
# variable constant. These values are vital in understanding the impact of each predictor on 'adiposity'.
# """
# print('Intercept: ', model.intercept_)
# print('Coefficients: ', model.coef_)
#
# # Plot original data and the model output
# """
# Visualizing the true vs. predicted values allows for an initial qualitative assessment of the model's performance.
# A perfect model would have all points lying on a 45-degree line (indicating true equals predicted).
# Although we've plotted against only one variable (Obesity) for simplicity, it's essential to note that 'Age' is also a
# predictor in the model.
# """
# plt.scatter(X_test[:, 0], y_test, marker='.', label='True')  # Test data
# plt.scatter(X_test[:, 0], y_est, color='r', marker='.', label='Predicted')  # Predicted data
# plt.xlabel('Obesity and Age (scaled)')
# plt.ylabel('Adiposity')
# plt.legend()
# plt.title('Adiposity vs. Obesity with Age (as a predictor)')
# plt.show()
#
# """
# 2. Introduce a regularization parameter λ as discussed in chapter 14 of the lecture
# notes, and estimate the generalization error for different values of λ. Specifi-
# cally, choose a reasonable range of values of λ (ideally one where the general-
# ization error first drop and then increases), and for each value use K = 10 fold
# cross-validation (algorithm 5) to estimate the generalization error.
# Include a figure of the estimated generalization error as a function of λ in the
# report and briefly discuss the result.
# """
# import numpy as np
# import pandas as pd
# from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, title, subplot, show, grid)
# from sklearn.model_selection import KFold
# from Tools.toolbox_02450 import rlr_validate
# import sklearn.linear_model as lm
#
# # # Load csv file with data
# # filename = 'saheart_1_withheader.csv'
# # data = pd.read_csv(filename)
# #
# # # Extract 'obesity' (target variable) and 'adiposity' (predictor variable)
# # y = data['obesity'].values
# # X = data[['adiposity']].values
#
# # Add offset attribute
# X = np.concatenate((np.ones((X.shape[0],1)),X),1)
# attributeNames = ['Offset', 'obesity', 'age']
# M = X.shape[1]
#
# ## Crossvalidation
# # Create crossvalidation partition for evaluation
# K = 10
# CV = KFold(K, shuffle=True)
#
# # Values of lambda
# lambdas = np.power(10.,range(-5,9))
#
# # Initialize variables
# Error_train = np.empty((K,1))
# Error_test = np.empty((K,1))
# Error_train_rlr = np.empty((K,1))
# Error_test_rlr = np.empty((K,1))
# Error_train_nofeatures = np.empty((K,1))
# Error_test_nofeatures = np.empty((K,1))
# w_rlr = np.empty((M,K))
# mu = np.empty((K, M-1))
# sigma = np.empty((K, M-1))
# w_noreg = np.empty((M,K))
#
# k = 0
# for train_index, test_index in CV.split(X, y):
#
#     # extract training and test set for current CV fold
#     X_train = X[train_index]
#     y_train = y[train_index]
#     X_test = X[test_index]
#     y_test = y[test_index]
#     internal_cross_validation = 10
#
#     opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, y_train,
#                                                                                                       lambdas,
#                                                                                                       internal_cross_validation)
#
#     # Standardize outer fold based on training set, and save the mean and standard
#     # deviations since they're part of the model (they would be needed for
#     # making new predictions) - for brevity we won't always store these in the scripts
#     mu[k, :] = np.mean(X_train[:, 1:], 0)
#     sigma[k, :] = np.std(X_train[:, 1:], 0)
#
#     X_train[:, 1:] = (X_train[:, 1:] - mu[k, :]) / sigma[k, :]
#     X_test[:, 1:] = (X_test[:, 1:] - mu[k, :]) / sigma[k, :]
#
#     Xty = X_train.T @ y_train
#     XtX = X_train.T @ X_train
#
#     # Compute mean squared error without using the input data at all
#     Error_train_nofeatures[k] = np.square(y_train - y_train.mean()).sum(axis=0) / y_train.shape[0]
#     Error_test_nofeatures[k] = np.square(y_test - y_test.mean()).sum(axis=0) / y_test.shape[0]
#
#     # Estimate weights for the optimal value of lambda, on entire training set
#     lambdaI = opt_lambda * np.eye(M)
#     lambdaI[0, 0] = 0  # Do no regularize the bias term
#     w_rlr[:, k] = np.linalg.solve(XtX + lambdaI, Xty).squeeze()
#     # Compute mean squared error with regularization with optimal lambda
#     Error_train_rlr[k] = np.square(y_train - X_train @ w_rlr[:, k]).sum(axis=0) / y_train.shape[0]
#     Error_test_rlr[k] = np.square(y_test - X_test @ w_rlr[:, k]).sum(axis=0) / y_test.shape[0]
#
#     # Estimate weights for unregularized linear regression, on entire training set
#     w_noreg[:, k] = np.linalg.solve(XtX, Xty).squeeze()
#     # Compute mean squared error without regularization
#     Error_train[k] = np.square(y_train - X_train @ w_noreg[:, k]).sum(axis=0) / y_train.shape[0]
#     Error_test[k] = np.square(y_test - X_test @ w_noreg[:, k]).sum(axis=0) / y_test.shape[0]
#     # OR ALTERNATIVELY: you can use sklearn.linear_model module for linear regression:
#     # m = lm.LinearRegression().fit(X_train, y_train)
#     # Error_train[k] = np.square(y_train-m.predict(X_train)).sum()/y_train.shape[0]
#     # Error_test[k] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]
#
#     # Display the results for the last cross-validation fold
#     if k == K - 1:
#         figure(k, figsize=(12, 8))
#         subplot(1, 2, 1)
#         semilogx(lambdas, mean_w_vs_lambda.T[:, 1:], '.-')  # Don't plot the bias term
#         xlabel('Regularization factor')
#         ylabel('Mean Coefficient Values')
#         grid()
#         # You can choose to display the legend, but it's omitted for a cleaner
#         # plot, since there are many attributes
#         # legend(attributeNames[1:], loc='best')
#
#         subplot(1, 2, 2)
#         title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
#         loglog(lambdas, train_err_vs_lambda.T, 'b.-', lambdas, test_err_vs_lambda.T, 'r.-')
#         xlabel('Regularization factor')
#         ylabel('Squared error (crossvalidation)')
#         legend(['Train error', 'Validation error'])
#         grid()
#
#     # To inspect the used indices, use these print statements
#     # print('Cross validation fold {0}/{1}:'.format(k+1,K))
#     # print('Train indices: {0}'.format(train_index))
#     # print('Test indices: {0}\n'.format(test_index))
#
#     k += 1
#
# # Display results
# print('Linear regression without feature selection:')
# print('- Training error: {0}'.format(Error_train.mean()))
# print('- Test error:     {0}'.format(Error_test.mean()))
# print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum() - Error_train.sum()) / Error_train_nofeatures.sum()))
# print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum() - Error_test.sum()) / Error_test_nofeatures.sum()))
# print('Regularized linear regression:')
# print('- Training error: {0}'.format(Error_train_rlr.mean()))
# print('- Test error:     {0}'.format(Error_test_rlr.mean()))
# print('- R^2 train:     {0}'.format(
#     (Error_train_nofeatures.sum() - Error_train_rlr.sum()) / Error_train_nofeatures.sum()))
# print(
#     '- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum() - Error_test_rlr.sum()) / Error_test_nofeatures.sum()))
#
# print('Weights in last fold:')
# for m in range(M):
#     print('{:>15} {:>15}'.format(attributeNames[m], np.round(w_rlr[m, -1], 2)))
#
# show()
# print('From Exercise 8.1.1')
#
#
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.linear_model import Ridge
# from sklearn.model_selection import cross_val_score, KFold
# from sklearn.metrics import mean_squared_error
#
# # # Load csv file with data
# # filename = 'saheart_1_withheader.csv'
# # data = pd.read_csv(filename)
# #
# # # Extract 'obesity' (target variable) and 'adiposity' (predictor variable)
# # y = data['obesity'].values
# # X = data[['adiposity']].values
#
# y = data['adiposity'].values
# X = data[['obesity', 'age']].values
#
# # Standardize the input
# X_scaled = (X - np.mean(X)) / np.std(X)
#
# # Set up cross-validation
# K = 10
# kfold = KFold(n_splits=K)
#
# # Introduce a regularization parameter λ
# lambda_interval = np.logspace(-8, 2, 50)
# mse_values = np.zeros(len(lambda_interval))
#
# # Perform cross-validation for each λ
# for i, λ in enumerate(lambda_interval):
#     ridge = Ridge(alpha=λ)
#     mse = -cross_val_score(ridge, X_scaled, y, cv=kfold, scoring='neg_mean_squared_error').mean()
#     mse_values[i] = mse
#
# # Find the optimal λ
# opt_lambda_idx = np.argmin(mse_values)
# opt_lambda = lambda_interval[opt_lambda_idx]
# min_error = mse_values[opt_lambda_idx]
#
# print(f"Optimal Lambda Index: {opt_lambda_idx}")
# print(f"Optimal Lambda Value: {opt_lambda}")
# print(f"Minimum Error (MSE) with Optimal Lambda: {min_error}")
#
# # Plot generalization error as a function of λ
# plt.figure(figsize=(8, 8))
# plt.semilogx(lambda_interval, mse_values)
# plt.semilogx(opt_lambda, min_error, 'o')
# plt.text(1e-8, min_error + 0.1, f"Minimum MSE: {min_error:.2f} at λ = {opt_lambda:.2e}")
# plt.xlabel('Regularization strength, λ')
# plt.ylabel('Mean squared error')
# plt.title('Generalization error as a function of λ')
# plt.legend(['MSE', 'Optimal λ'], loc='upper right')
# plt.grid()
# plt.show()
#
# print('From Exercise 8.1.2')

"""
3. Explain how a new data observation is predicted according to the linear model
with the lowest generalization error as estimated in the previous question. I.e.,
what are the effects of the selected attributes in terms of determining the
predicted class. Does the result make sense?
"""

"""
Regression, part b: In this section, we will compare three models: the regularized
linear regression model from the previous section, an artificial neural network (ANN)
and a baseline. We are interested in two questions: Is one model better than the
other? Is either model better than a trivial baseline?. We will attempt to answer
these questions with two-level cross-validation.

1- Implement two-level cross-validation (see algorithm 6 of the lecture notes). We
will use 2-level cross-validation to compare the models with K1 = K2 = 10
folds4. As a baseline model, we will apply a linear regression model with no
features, i.e. it computes the mean of y on the training data, and use this value
to predict y on the test data.
Make sure you can fit an ANN model to the data. As complexity-controlling
parameter for the ANN, we will use the number of hidden units5 h. Based on
a few test-runs, select a reasonable range of values for h (which should include
h = 1), and describe the range of values you will use for h and λ
"""
"""
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LinearRegression, Ridge
from Tools.toolbox_02450 import train_neural_net, visualize_decision_boundary, draw_neural_net, rlr_validate
import torch
from sklearn.metrics import mean_squared_error

# Load csv file with data
filename = 'saheart_1_withheader.csv'
data = pd.read_csv(filename)

# # Extract 'obesity' (target variable) and 'adiposity' (predictor variable)
# y = data['obesity'].values[:, np.newaxis]
# X = data[['adiposity']].values


y = data['adiposity'].values[:, np.newaxis]
X = data[['obesity', 'age']].values


# K1 and K2 fold CrossValidation
K1 = 10
K2 = 10
CV_outer = model_selection.KFold(K1, shuffle=True)
CV_inner = model_selection.KFold(K2, shuffle=True)

# Loss function
loss_fn = torch.nn.BCELoss()

# Range of values for h and λ
hidden_units_range = [1, 2, 5, 10, 20]
lambdas = np.logspace(-5, 2, 10)

# Two-level cross-validation
errors = []  # To store the errors for each fold

# Outer cross-validation loop
# Used for evaluating the generalization error of the model
for k1, (train_index_outer, test_index_outer) in enumerate(CV_outer.split(X, y)):
    # Splitting the data into training and test sets for outer fold
    X_train_outer = X[train_index_outer]
    y_train_outer = y[train_index_outer]
    X_test_outer = X[test_index_outer]
    y_test_outer = y[test_index_outer]

    val_errors_ann = []  # Store validation errors for ANN model across different hyperparameters
    val_errors_linreg = []  # Store validation errors for Linear Regression model across different lambda

    # Iterating through potential hidden layer sizes for ANN
    for h in hidden_units_range:
        inner_errors_ann = []  # Store inner loop errors for ANN model

        # Inner cross-validation loop for ANN model
        # Used for selecting the hyperparameters of the model
        for k2, (train_index_inner, test_index_inner) in enumerate(CV_inner.split(X_train_outer, y_train_outer)):
            # Splitting the data into training and test sets for inner fold
            X_train_inner = X[train_index_inner]
            y_train_inner = y[train_index_inner]
            X_test_inner = X[test_index_inner]
            y_test_inner = y[test_index_inner]

            # Define the ANN model structure
            model = lambda: torch.nn.Sequential(
                torch.nn.Linear(X_train_inner.shape[1], h),
                torch.nn.Tanh(),
                torch.nn.Linear(h, 1),
                torch.nn.Sigmoid()
            )

            # Train the ANN model
            # Assumed external function, details not provided
            net, _, _ = train_neural_net(model,
                                         loss_fn,
                                         X=torch.Tensor(X_train_inner),
                                         y=torch.Tensor(y_train_inner),
                                         n_replicates=1,
                                         max_iter=10)

            # Evaluate ANN model on the inner test set and store the error
            y_pred_inner_ann = net(torch.Tensor(X_test_inner))
            error_inner_ann = mean_squared_error(y_test_inner, y_pred_inner_ann.data.numpy().squeeze())
            inner_errors_ann.append(error_inner_ann)

        # Calculate and store the average error across all inner folds for a specific ANN model
        val_errors_ann.append(np.mean(inner_errors_ann))

    # Iterating through potential lambda values for Linear Regression
    for lambda_ in lambdas:
        inner_errors_linreg = []  # Store inner loop errors for Linear Regression model

        # Inner cross-validation loop for Linear Regression model
        for k2, (train_index_inner, test_index_inner) in enumerate(CV_inner.split(X_train_outer, y_train_outer)):
            # Splitting the data into training and test sets for inner fold
            X_train_inner = X[train_index_inner]
            y_train_inner = y[train_index_inner]
            X_test_inner = X[test_index_inner]
            y_test_inner = y[test_index_inner]

            # Train Linear Regression model with regularization (Ridge Regression)
            linreg_model = Ridge(alpha=lambda_).fit(X_train_inner, y_train_inner)

            # Evaluate Linear Regression model on the inner test set and store the error
            y_pred_inner_linreg = linreg_model.predict(X_test_inner)
            error_inner_linreg = mean_squared_error(y_test_inner, y_pred_inner_linreg)
            inner_errors_linreg.append(error_inner_linreg)

        # Calculate and store the average error across all inner folds for a specific lambda
        val_errors_linreg.append(np.mean(inner_errors_linreg))

    # Determine the best hyperparameters for ANN and Linear Regression model based on validation errors
    best_h = hidden_units_range[np.argmin(val_errors_ann)]
    best_lambda = lambdas[np.argmin(val_errors_linreg)]

    # Define and train the best ANN model using outer training data
    best_ann_model = lambda: torch.nn.Sequential(
        torch.nn.Linear(X_train_outer.shape[1], best_h),
        torch.nn.Tanh(),
        torch.nn.Linear(best_h, 1),
        torch.nn.Sigmoid()
    )
    net, _, _ = train_neural_net(best_ann_model,
                                 loss_fn,
                                 X=torch.Tensor(X_train_outer),
                                 y=torch.Tensor(y_train_outer),
                                 n_replicates=1,
                                 max_iter=10)

    # Train the best Linear Regression model using outer training data
    best_linreg_model = Ridge(alpha=best_lambda).fit(X_train_outer, y_train_outer)

    # Evaluate the best ANN and Linear Regression models on the outer test data
    y_pred_outer_ann = net(torch.Tensor(X_test_outer))
    error_outer_ann = mean_squared_error(y_test_outer, y_pred_outer_ann.data.numpy().squeeze())

    y_pred_outer_linreg = best_linreg_model.predict(X_test_outer)
    error_outer_linreg = mean_squared_error(y_test_outer, y_pred_outer_linreg)

    # Store the errors for each model and print the results of each outer fold
    errors.append((error_outer_linreg, error_outer_ann))

    print(
        f'Fold {k1 + 1}/{K1}: LinReg error = {error_outer_linreg:.4f}, ANN error = {error_outer_ann:.4f}, Best h = {best_h}, Best lambda = {best_lambda:.5f}')

# Calculate and print the average test error over all outer folds for both models
average_errors = np.mean(errors, axis=0)
print(f'Average LinReg error = {average_errors[0]:.4f}, Average ANN error = {average_errors[1]:.4f}')

# Fold 10/10: LinReg error = 16.5851, ANN error = 715.2919, Best h = 20, Best lambda = 0.07743
# Average LinReg error = 18.8231, Average ANN error = 684.751

print('Exercises 8.1.1, 8.1.2 and 8.2.2.')
"""
"""
2. Produce a table akin to Table 1 using two-level cross-validation (algorithm 6
in the lecture notes). The table shows, for each of the K1 = 10 folds i, the
optimal value of the number of hidden units and regularization strength (h∗
i and λ∗i respectively) as found after each inner loop, as well as the estimated
generalization errors Etest i by evaluating on Dtest
i . It also includes the baseline test error, also evaluated on Dtest
i . Importantly, you must re-use the train/test
splits Dpar i , Dtest
i for all three methods to allow statistical comparison (see next section).
Note the error measure we use is the squared loss per observation, i.e. we divide
by the number of observation in the test dataset:
E = 1
N test
N test
∑i=1
(yi − ˆyi)2
Include a table similar to Table 1 in your report and briefly discuss what it tells
you at a glance. Do you find the same value of λ∗ as in the previous section?
"""
"""
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LinearRegression, Ridge
from Tools.toolbox_02450 import train_neural_net, visualize_decision_boundary, draw_neural_net, rlr_validate
import torch
from sklearn.metrics import mean_squared_error

# Load csv file with data
filename = 'saheart_1_withheader.csv'
data = pd.read_csv(filename)

# Extract 'obesity' (target variable) and 'adiposity' (predictor variable)
y = data['obesity'].values
y = y[:, np.newaxis]
X = data[['adiposity']].values

y = data['adiposity'].values[:, np.newaxis]
X = data[['obesity', 'age']].values

# K1 and K2 fold CrossValidation
K1 = 10
K2 = 10
CV_outer = model_selection.KFold(K1, shuffle=True)
CV_inner = model_selection.KFold(K2, shuffle=True)

# Loss function for ANN
loss_fn = torch.nn.BCELoss()

# Range of values for h and λ
hidden_units_range = [1, 2, 5, 10, 20]
lambdas = np.logspace(-5, 2, 10)

# Initialize lists to store errors and best parameters
errors = []
best_params = []

# Outer cross-validation loop
for k1, (train_index_outer, test_index_outer) in enumerate(CV_outer.split(X, y)):
    X_train_outer = X[train_index_outer]
    y_train_outer = y[train_index_outer]
    X_test_outer = X[test_index_outer]
    y_test_outer = y[test_index_outer]

    # Linear Regression and ANN model validation error storage
    val_errors_ann = []
    val_errors_linreg = []

    # Inner cross-validation loop for ANN model
    for h in hidden_units_range:
        inner_errors_ann = []
        for k2, (train_index_inner, test_index_inner) in enumerate(CV_inner.split(X_train_outer, y_train_outer)):
            X_train_inner = X[train_index_inner]
            y_train_inner = y[train_index_inner]
            X_test_inner = X[test_index_inner]
            y_test_inner = y[test_index_inner]

            # Define the ANN model structure
            model = lambda: torch.nn.Sequential(
                torch.nn.Linear(X_train_inner.shape[1], h),
                torch.nn.Tanh(),
                torch.nn.Linear(h, 1),
                torch.nn.Sigmoid()
            )

            # Train the ANN model
            # Assumed external function, details not provided
            net, _, _ = train_neural_net(model,
                                         loss_fn,
                                         X=torch.Tensor(X_train_inner),
                                         y=torch.Tensor(y_train_inner),
                                         n_replicates=1,
                                         max_iter=10)

            # Evaluate ANN model on the inner test set and store the error
            y_pred_inner_ann = net(torch.Tensor(X_test_inner))
            error_inner_ann = mean_squared_error(y_test_inner, y_pred_inner_ann.data.numpy().squeeze())

            # Store error for this inner fold
            inner_errors_ann.append(error_inner_ann)

        # Store average validation error for this 'h'
        val_errors_ann.append(np.mean(inner_errors_ann))

    # Inner cross-validation loop for Linear Regression model
    for lambda_ in lambdas:
        inner_errors_linreg = []
        for k2, (train_index_inner, test_index_inner) in enumerate(CV_inner.split(X_train_outer, y_train_outer)):
            X_train_inner = X[train_index_inner]
            y_train_inner = y[train_index_inner]
            X_test_inner = X[test_index_inner]
            y_test_inner = y[test_index_inner]

            # Define, train, and evaluate the Linear Regression model
            # Train Linear Regression model with regularization (Ridge Regression)
            linreg_model = Ridge(alpha=lambda_).fit(X_train_inner, y_train_inner)

            # Evaluate Linear Regression model on the inner test set and store the error
            y_pred_inner_linreg = linreg_model.predict(X_test_inner)
            error_inner_linreg = mean_squared_error(y_test_inner, y_pred_inner_linreg)

            # Store error for this inner fold
            inner_errors_linreg.append(error_inner_linreg)

        # Store average validation error for this 'lambda_'
        val_errors_linreg.append(np.mean(inner_errors_linreg))

    # Determine the best hyperparameters for ANN and Linear Regression model based on validation errors
    best_h = hidden_units_range[np.argmin(val_errors_ann)]
    best_lambda = lambdas[np.argmin(val_errors_linreg)]

    # Train models with best parameters on the entire outer training set and evaluate on the outer test set
    # Define and train the best ANN model using outer training data
    best_ann_model = lambda: torch.nn.Sequential(
        torch.nn.Linear(X_train_outer.shape[1], best_h),
        torch.nn.Tanh(),
        torch.nn.Linear(best_h, 1),
        torch.nn.Sigmoid()
    )
    net, _, _ = train_neural_net(best_ann_model,
                                 loss_fn,
                                 X=torch.Tensor(X_train_outer),
                                 y=torch.Tensor(y_train_outer),
                                 n_replicates=1,
                                 max_iter=10)

    # Train the best Linear Regression model using outer training data
    best_linreg_model = Ridge(alpha=best_lambda).fit(X_train_outer, y_train_outer)

    # Evaluate the best ANN and Linear Regression models on the outer test data
    y_pred_outer_ann = net(torch.Tensor(X_test_outer))
    error_outer_ann = mean_squared_error(y_test_outer, y_pred_outer_ann.data.numpy().squeeze())

    y_pred_outer_linreg = best_linreg_model.predict(X_test_outer)
    error_outer_linreg = mean_squared_error(y_test_outer, y_pred_outer_linreg)

    # Store the errors and best parameters
    errors.append((error_outer_linreg, error_outer_ann))
    best_params.append((best_h, best_lambda))

    print(
        f'Fold {k1 + 1}/{K1}: LinReg error = {error_outer_linreg:.4f}, ANN error = {error_outer_ann:.4f}, Best h = {best_h}, Best lambda = {best_lambda:.5f}')

# Calculate and print the average test error over all outer folds for both models
average_errors = np.mean(errors, axis=0)
print(f'Average LinReg error = {average_errors[0]:.4f}, Average ANN error = {average_errors[1]:.4f}')

# Create and display a table with the results
results_table = pd.DataFrame(best_params, columns=["Best h", "Best lambda"])
results_table["LinReg Error"] = [e[0] for e in errors]
results_table["ANN Error"] = [e[1] for e in errors]

print(results_table)

print('Exercises 8.1.1, 8.1.2 and 8.2.2.')
"""
"""
id  Best h  Best lambda  Baseline Error  Best ANN Error
0      10   100.000000       22.930121      728.698557
1      20   100.000000       11.557204      696.582861
2       2     2.782559        5.463190      661.715623
3      20   100.000000        4.875189      645.083560
4       1   100.000000       11.216089      693.967698
5       5     0.000359        6.680161      658.948975
6       1     0.464159        6.352755      653.861820
7       5     0.464159        4.036484      682.313379
8      20     0.077426        6.261912      640.655117
9      20     0.464159        7.209111      654.255880
"""



"""
3. Statistically evaluate if there is a significant performance difference between the
fitted ANN, linear regression model and baseline using the methods described
in chapter 11. These comparisons will be made pairwise (ANN vs. linear
regression; ANN vs. baseline; linear regression vs. baseline). We will allow
some freedom in what test to choose. Therefore, choose either:

"""
"""
setup I (section 11.3): Use the paired t-test described in Box 11.3.4
setup II (section 11.4): Use the method described in Box 11.4.1)
4If this is too time-consuming, use K1 = K2 = 5
5Note there are many things we could potentially tweak or select, such as regularization. If you
wish to select another parameter to tweak feel free to do so.
"""
"""

Outer fold ANN Linear regression baseline
i h∗-i Etest-i λ∗-i Etest-i Etest-i
1 3    10.8    0.01 12.8    15.3
2 4    10.1    0.01 12.4    15.1
...    ...     ...  ...     ...
10 3   10.9    0.05 12.1    15.9
Table 1: Two-level cross-validation table used to compare the three models
Include p-values and confidence intervals for the three pairwise tests in your
report and conclude on the results: Is one model better than the other? Are
the two models better than the baseline? Are some of the models identical?
What recommendations would you make based on what you’ve learned?
"""

"""
11.3.4 -> 7_2_1
"""

"""
11.4.1 -> 7_3_1 
"""
# """
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LinearRegression, Ridge
from Tools.toolbox_02450 import train_neural_net, visualize_decision_boundary, draw_neural_net
import torch

# Load csv file with data
filename = 'saheart_1_withheader.csv'
data = pd.read_csv(filename)

# Extract 'obesity' (target variable) and 'adiposity' (predictor variable)
y = data['obesity'].values
X = data[['adiposity']].values


y = data['adiposity'].values[:, np.newaxis]
X = data[['obesity', 'age']].values


# Define the number of outer and inner folds
K1 = 10
K2 = 10

# Initialize the outer loop cross-validation
CV1 = model_selection.KFold(n_splits=K1, shuffle=True)

# Initialize the results lists
r_ann_vs_lr = []
r_ann_vs_baseline = []
r_lr_vs_baseline = []

best_net = None

# Outer loop
for (outer_train_index, outer_test_index) in CV1.split(X, y):
    X_train_outer, y_train_outer = X[outer_train_index], y[outer_train_index]
    X_test_outer, y_test_outer = X[outer_test_index], y[outer_test_index]

    # Inner loop cross-validation
    CV2 = model_selection.KFold(n_splits=K2, shuffle=True)

    # Initialize inner loop results lists
    ann_errors = []
    lr_errors = []
    baseline_errors = []

    # Inner loop
    for (inner_train_index, inner_test_index) in CV2.split(X_train_outer, y_train_outer):
        X_train_inner, y_train_inner = X_train_outer[inner_train_index], y_train_outer[inner_train_index]
        X_test_inner, y_test_inner = X_train_outer[inner_test_index], y_train_outer[inner_test_index]


        loss_fn = torch.nn.MSELoss()
        # Train the ANN
        hidden_units = 10

        model = lambda: torch.nn.Sequential(
            torch.nn.Linear(1, hidden_units),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_units, 1),
            torch.nn.Sigmoid()
        )
        # Update this line to pass the model and loss_fn as arguments
        best_net, _, _ = train_neural_net(model,
                         loss_fn,
                         X=torch.Tensor(X_train_inner),
                         y=torch.Tensor(y_train_inner),
                         n_replicates=1,
                         max_iter=100)

        # Calculate the ANN error
        y_pred_inner = best_net[0](torch.Tensor(X_test_inner)).detach().numpy()
        ann_error = np.mean((y_pred_inner - y_test_inner.reshape(-1, 1)) ** 2)

        ann_errors.append(ann_error)

        # Train the linear regression model
        lr_model = LinearRegression().fit(X_train_inner, y_train_inner)

        # Calculate the linear regression error
        lr_error = np.mean((lr_model.predict(X_test_inner) - y_test_inner) ** 2)
        lr_errors.append(lr_error)

        # Calculate the baseline error
        baseline_error = np.mean((np.mean(y_train_inner) - y_test_inner) ** 2)
        baseline_errors.append(baseline_error)

    # Calculate the average errors for the inner loop
    avg_ann_error = np.mean(ann_errors)
    avg_lr_error = np.mean(lr_errors)
    avg_baseline_error = np.mean(baseline_errors)


    # Calculate the outer loop errors
    y_pred_outer = best_net[0](torch.Tensor(X_test_outer)).detach().numpy()
    ann_outer_error = np.mean((y_pred_outer - y_test_outer.reshape(-1, 1)) ** 2)
    lr_outer_error = np.mean((lr_model.predict(X_test_outer) - y_test_outer) ** 2)
    baseline_outer_error = np.mean((np.mean(y_train_outer) - y_test_outer) ** 2)

    # Append the differences in performance to the results lists
    r_ann_vs_lr.append(ann_outer_error - lr_outer_error)
    r_ann_vs_baseline.append(ann_outer_error - baseline_outer_error)
    r_lr_vs_baseline.append(lr_outer_error - baseline_outer_error)


# Perform statistical evaluation using the correlated t-test (setup II)
alpha = 0.05
rho = 1 / K1

p_ann_vs_lr, CI_ann_vs_lr = correlated_ttest(r_ann_vs_lr, rho, alpha=alpha)
p_ann_vs_baseline, CI_ann_vs_baseline = correlated_ttest(r_ann_vs_baseline, rho, alpha=alpha)
p_lr_vs_baseline, CI_lr_vs_baseline = correlated_ttest(r_lr_vs_baseline, rho, alpha=alpha)

# Print the results
print("ANN vs Linear Regression")
print(f"Confidence interval: {CI_ann_vs_lr}")
print(f"p-value: {p_ann_vs_lr}\n")

print("ANN vs Baseline")
print(f"Confidence interval: {CI_ann_vs_baseline}")
print(f"p-value: {p_ann_vs_baseline}\n")

print("Linear Regression vs Baseline")
print(f"Confidence interval: {CI_lr_vs_baseline}")
print(f"p-value: {p_lr_vs_baseline}\n")

print('Exercises 7.2.1 and 7.3.1.')

# """
"""
ANN vs Linear Regression
Confidence interval: (618.9158995001742, 1065.1464151787093)
p-value: 1.3120354154448714e-05
ANN vs Baseline
Confidence interval: (610.6730468456259, 1055.0686004229283)
p-value: 1.386470926010315e-05
Linear Regression vs Baseline
Confidence interval: (-11.939874404930247, -6.380793005399369)
p-value: 3.870749042494397e-0
"""

"""
Classification: In this part of the report you are to solve a relevant classification
problem for your data and statistically evaluate your result. The tasks will closely
mirror what you just did in the last section. The three methods we will compare is a
baseline, logistic regression, and one of the other four methods from below (referred
to as method 2 ).
Logistic regression for classification. Once more, we can use a regularization pa-
rameter λ ≥ 0 to control complexity
ANN Artificial neural networks for classification. Same complexity-controlling pa-
rameter as in the previous exercise
CT Classification trees. Same complexity-controlling parameter as for regression
trees
KNN k-nearest neighbor classification, complexity controlling parameter k = 1, 2 . . .
NB Na ̈ıve Bayes. As complexity-controlling parameter, we suggest the term b ≥ 0
from section 11.2.1 of the lecture notes to estimate6 p(x = 1) = n++b
n++n−+2b

1. Explain which classification problem you have chosen to solve. Is it a multi-
class or binary classification problem?


KFold: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
KNeighborsClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
LogisticRegression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
accuracy_score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
https://medium.com/nerd-for-tech/cross-validation-6270341658ae
"""


"""
2. We will compare logistic regression, method 2 and a baseline. For logistic
regression, we will once more use λ as a complexity-controlling parameter, and
for method 2 a relevant complexity controlling parameter and range of values.
We recommend this choice is made based on a trial run, which you do not need
to report. Describe which parameter you have chosen and the possible values
of the parameters you will examine.
The baseline will be a model which compute the largest class on the training
data, and predict everything in the test-data as belonging to that class (corre-
sponding to the optimal prediction by a logistic regression model with a bias
term and no features).
"""
"""
import pandas as pd
import numpy as np
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

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from scipy import stats

# Split data into training and test sets
test_proportion = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_proportion)

# Logistic Regression with lambda as complexity-controlling parameter
lmbda = 0.1 # adjust the value of lambda as needed
logistic_model = LogisticRegression(solver='lbfgs', max_iter=1000, C=1/lmbda)
logistic_model.fit(X_train, y_train)
yhat_logistic = logistic_model.predict(X_test)

# K-Nearest Neighbors Classifier (Method 2)
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
yhat_knn = knn_model.predict(X_test)

# Baseline Model
# Baseline Model
majority_class = stats.mode(y_train)[0][0]
yhat_baseline = np.full_like(y_test, majority_class)

# Calculate accuracy scores
logistic_accuracy = accuracy_score(y_test, yhat_logistic)
knn_accuracy = accuracy_score(y_test, yhat_knn)
baseline_accuracy = accuracy_score(y_test, yhat_baseline)

print("Logistic Regression Accuracy:", logistic_accuracy)
print("K-Nearest Neighbors Accuracy:", knn_accuracy)
print("Baseline Model Accuracy:", baseline_accuracy)

# Logistic Regression Accuracy: 0.7741935483870968
# K-Nearest Neighbors Accuracy: 0.6129032258064516
# Baseline Model Accuracy: 0.6451612903225806

print('Similar to Exercise 7.2.1')
"""
"""
3. Again use two-level cross-validation to create a table similar to Table 2, but
now comparing the logistic regression, method 2, and baseline. The table should
once more include the selected parameters, and as an error measure we will use
the error rate:
E = {Number of misclassified observations}/N^test
Once more, make sure to re-use the outer validation splits to admit statistical
evaluation. Briefly discuss the result.
"""
"""
import numpy as np
from sklearn.model_selection import KFold

# K-fold CrossValidation
K = 10
CV = KFold(n_splits=K, shuffle=True)

# Range of values for k-nearest neighbors and λ
L=40
lambdas = np.power(10.,range(-5,9))


results = []

for train_index, test_index in CV.split(X):
    X_train = X[train_index, :]
    y_train = y[train_index]
    X_test = X[test_index, :]
    y_test = y[test_index]

    best_knn_accuracy = 0
    best_k = 0
    for k in range(1, L + 1):
        knn_model = KNeighborsClassifier(n_neighbors=k)
        knn_model.fit(X_train, y_train.ravel())
        yhat_knn = knn_model.predict(X_test)
        knn_accuracy = accuracy_score(y_test, yhat_knn)
        if knn_accuracy > best_knn_accuracy:
            best_knn_accuracy = knn_accuracy
            best_k = k

    best_logistic_accuracy = 0
    best_lambda = 0
    for lmbda in lambdas:
        logistic_model = LogisticRegression(solver='lbfgs', max_iter=1000, C=1 / lmbda)
        logistic_model.fit(X_train, y_train.ravel())
        yhat_logistic = logistic_model.predict(X_test)
        logistic_accuracy = accuracy_score(y_test, yhat_logistic)
        if logistic_accuracy > best_logistic_accuracy:
            best_logistic_accuracy = logistic_accuracy
            best_lambda = lmbda

    majority_class = stats.mode(y_train)[0][0]
    yhat_baseline = np.full_like(y_test, majority_class)
    baseline_accuracy = accuracy_score(y_test, yhat_baseline)

    results.append((best_k, 1 - best_knn_accuracy, best_lambda, 1 - best_logistic_accuracy, 1 - baseline_accuracy))

results_table = pd.DataFrame(results, columns=["Best k", "Etest_knn", "Best λ", "Etest_logistic", "Etest_baseline"])
print(results_table)

print('Similar to Exercise 6.2.1 and 6.3.2 with DecisionTreeClassifier')

"""
"""
   Best k  Etest_knn        Best λ  Etest_logistic  Etest_baseline
0       4   0.234043    1000.00000        0.234043        0.297872
1       2   0.319149  100000.00000        0.234043        0.234043
2       3   0.500000     100.00000        0.326087        0.478261
3       2   0.347826       0.00001        0.217391        0.369565
4       5   0.326087     100.00000        0.260870        0.304348
5       4   0.456522   10000.00000        0.347826        0.413043
6       2   0.347826   10000.00000        0.304348        0.326087
7       1   0.326087       0.00001        0.282609        0.434783
8       2   0.304348       0.00001        0.152174        0.304348
9       4   0.326087       0.00001        0.152174        0.304348
"""

"""
4. Perform a statistical evaluation of your three models similar to the previous
section. That is, compare the three models pairwise. We will once more allow
some freedom in what test to choose. Therefore, choose either:
setup I (section 11.3): Use McNemera’s test described in Box 11.3.2)
setup II (section 11.4): Use the method described in Box 11.4.1)
Include p-values and confidence intervals for the three pairwise tests in your
report and conclude on the results: Is one model better than the other? Are
the two models better than the baseline? Are some of the models identical?
What recommendations would you make based on what you’ve learned?
"""
"""
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import scipy.stats as st
from sklearn.model_selection import KFold

# Load csv file with data
filename = 'saheart_1_withheader.csv'
data = pd.read_csv(filename)

# Extract class names and encode with integers
classLabels = data.iloc[:, 0].values
classNames = sorted(set(classLabels))
classDict = dict(zip(classNames, range(2)))

# Extract vector y, convert to NumPy array
y = np.asarray([classDict[value] for value in classLabels])

# Extract matrix X
X = data.iloc[:, 1:10].values

# K-fold CrossValidation
K = 10
CV = KFold(n_splits=K, shuffle=True)

# Range of values for k-nearest neighbors and λ
L = 40
lambdas = np.power(10., range(-5, 9))

# Lists to store the best model predictions for each fold
best_knn_predictions = []
best_logistic_predictions = []
baseline_predictions = []

for train_index, test_index in CV.split(X):
    X_train = X[train_index, :]
    y_train = y[train_index]
    X_test = X[test_index, :]
    y_test = y[test_index]

    # Find the best KNN model
    best_knn_accuracy = 0
    best_yhat_knn = None
    for k in range(1, L + 1):
        knn_model = KNeighborsClassifier(n_neighbors=k)
        knn_model.fit(X_train, y_train.ravel())
        yhat_knn = knn_model.predict(X_test)
        knn_accuracy = accuracy_score(y_test, yhat_knn)
        if knn_accuracy > best_knn_accuracy:
            best_knn_accuracy = knn_accuracy
            best_yhat_knn = yhat_knn
    best_knn_predictions.append(best_yhat_knn)

    # Find the best logistic model
    best_logistic_accuracy = 0
    best_yhat_logistic = None
    for lmbda in lambdas:
        logistic_model = LogisticRegression(solver='lbfgs', max_iter=1000, C=1 / lmbda)
        logistic_model.fit(X_train, y_train.ravel())
        yhat_logistic = logistic_model.predict(X_test)
        logistic_accuracy = accuracy_score(y_test, yhat_logistic)
        if logistic_accuracy > best_logistic_accuracy:
            best_logistic_accuracy = logistic_accuracy
            best_yhat_logistic = yhat_logistic
    best_logistic_predictions.append(best_yhat_logistic)

    # Calculate the baseline model
    majority_class = stats.mode(y_train)[0][0]
    yhat_baseline = np.full_like(y_test, majority_class)
    baseline_predictions.append(yhat_baseline)

# Combine predictions across folds
best_knn_predictions = np.hstack(best_knn_predictions)
best_logistic_predictions = np.hstack(best_logistic_predictions)
baseline_predictions = np.hstack(baseline_predictions)

# Perform McNemar's test for all three pairwise comparisons
thetahat_knn_logistic, CI_knn_logistic, p_knn_logistic = mcnemar(y, best_knn_predictions, best_logistic_predictions, alpha=0.05)
thetahat_knn_baseline, CI_knn_baseline, p_knn_baseline = mcnemar(y, best_knn_predictions, baseline_predictions, alpha=0.05)
thetahat_logistic_baseline, CI_logistic_baseline, p_logistic_baseline = mcnemar(y, best_logistic_predictions, baseline_predictions, alpha=0.05)

# Print McNemar's test results
print("KNN vs Logistic Regression: theta_hat = {}, CI = {}, p-value = {}".format(thetahat_knn_logistic, CI_knn_logistic, p_knn_logistic))
print("KNN vs Baseline Model: theta_hat = {}, CI = {}, p-value = {}".format(thetahat_knn_baseline, CI_knn_baseline, p_knn_baseline))
print("Logistic Regression vs Baseline Model: theta_hat = {}, CI = {}, p-value = {}".format(thetahat_logistic_baseline, CI_logistic_baseline, p_logistic_baseline))

print('Similar to Exercise 7.1.1 and 7.1.4 with DecisionTreeClassifier and mcnemar')
"""

"""
KNN vs Logistic Regression: theta_hat = 0.006493506493506494, CI = (-0.036928157987507815, 0.04990309035848739), p-value = 0.8453703188825519
KNN vs Baseline Model: theta_hat = -0.06060606060606061, CI = (-0.1025654896097481, -0.018540669617686745), p-value = 0.006637120515926128
Logistic Regression vs Baseline Model: theta_hat = -0.0670995670995671, CI = (-0.1100255588521013, -0.02405064532611867), p-value = 0.0032209898687985275
"""


"""
5. Train a logistic regression model using a suitable value of λ (see previous ex-
ercise). Explain how the logistic regression model make a prediction. Are the
same features deemed relevant as for the regression part of the report?
"""

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load csv file with data
filename = 'saheart_1_withheader.csv'
data = pd.read_csv(filename)

# Extract class names and encode with integers
classLabels = data.iloc[:, 0].values
classNames = sorted(set(classLabels))
classDict = dict(zip(classNames, range(2)))

# Extract vector y, convert to NumPy array
y = np.asarray([classDict[value] for value in classLabels])

# Extract matrix X
X = data.iloc[:, 1:10].values

# Create crossvalidation partition for evaluation
# using stratification and 95 pct. split between training and test
K = 20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.95, stratify=y)

# Standardize the training and set set based on training set mean and std
mu = np.mean(X_train, 0)
sigma = np.std(X_train, 0)

X_train = (X_train - mu) / sigma
X_test = (X_test - mu) / sigma

# Fit regularized logistic regression model to training data to predict
# the type of wine
lambda_interval = np.logspace(-8, 2, 50)
train_error_rate = np.zeros(len(lambda_interval))
test_error_rate = np.zeros(len(lambda_interval))
coefficient_norm = np.zeros(len(lambda_interval))
for k in range(0, len(lambda_interval)):
    mdl = LogisticRegression(penalty='l2', C=1 / lambda_interval[k])

    mdl.fit(X_train, y_train)

    y_train_est = mdl.predict(X_train).T
    y_test_est = mdl.predict(X_test).T

    train_error_rate[k] = np.sum(y_train_est != y_train) / len(y_train)
    test_error_rate[k] = np.sum(y_test_est != y_test) / len(y_test)

    w_est = mdl.coef_[0]
    coefficient_norm[k] = np.sqrt(np.sum(w_est ** 2))

min_error = np.min(test_error_rate)
opt_lambda_idx = np.argmin(test_error_rate)
opt_lambda = lambda_interval[opt_lambda_idx]

# Print minimum test error and optimal lambda
print("Minimum test error: ", np.round(min_error * 100, 2), "%")
print("Optimal lambda: 1e", np.round(np.log10(opt_lambda), 2))

# Print the model weights for the optimal lambda
opt_mdl = LogisticRegression(penalty='l2', C=1/opt_lambda)
opt_mdl.fit(X_train, y_train)
opt_weights = opt_mdl.coef_[0]
print("Model weights for optimal lambda: ", opt_weights)

# Print the L2 norms of the weights for each lambda
print("L2 norms for each lambda: ", coefficient_norm)
print('From Exercise 8.1.2')
"""