import math

import numpy as np


def calc_cond_probs(y1_B, y2_B, y3_B):
    # Calculate conditional probabilities given predicted values
    yhat_B = np.array([y1_B, y2_B, y3_B])
    probs = np.exp(-yhat_B) / np.sum(np.exp(-yhat_B))
    cond_probs = probs / np.sum(probs)
    return cond_probs


y1_A = 1.2 - 2.1 * (-1.4) + 3.2 * (2.6)
y2_A = 1.2 - 1.7 * (-1.4) + 2.9 * (2.6)
y3_A = 1.3 - 1.1 * (-1.4) + 2.2 * (2.6)
print("y1_A =", y1_A)
print("y2_A =", y2_A)
print("y3_A =", y3_A)
sum_exp_ˆy_A = np.exp(y1_A) + np.exp(y2_A) + np.exp(y3_A)
print(sum_exp_ˆy_A)
cond_probs = calc_cond_probs(y1_A, y2_A, y3_A)
print(cond_probs)

y1_B = 1.2 - 2.1 * (-0.6) + 3.2 * (-1.6)
y2_B = 1.2 - 1.7 * (-0.6) + 2.9 * (-1.6)
y3_B = 1.3 - 1.1 * (-0.6) + 2.2 * (-1.6)
print("y1_B =", y1_B)
print("y2_B =", y2_B)
print("y3_B =", y3_B)
sum_exp_ˆy_B = np.exp(y1_B) + np.exp(y2_B) + np.exp(y3_B)
print(sum_exp_ˆy_B)
cond_probs = calc_cond_probs(y1_B, y2_B, y3_B)
print(cond_probs)

y1_C = 1.2 - 2.1 * 2.1 + 3.2 * 5
y2_C = 1.2 - 1.7 * 2.1 + 2.9 * 5
y3_C = 1.3 - 1.1 * 2.1 + 2.2 * 5
print("y1_C =", y1_C)
print("y2_C =", y2_C)
print("y3_C =", y3_C)
sum_exp_ˆy_C = np.exp(y1_C) + np.exp(y2_C) + np.exp(y3_C)
print(sum_exp_ˆy_C)
cond_probs = calc_cond_probs(y1_C, y2_C, y3_C)
print(cond_probs)

y1_D = 1.2 - 2.1 * (0.7) + 3.2 * (3.8)
y2_D = 1.2 - 1.7 * (0.7) + 2.9 * (3.8)
y3_D = 1.3 - 1.1 * (0.7) + 2.2 * (3.8)
print("y1_D =", y1_D)
print("y2_D =", y2_D)
print("y3_D =", y3_D)
sum_exp_ˆy_D = np.exp(y1_D) + np.exp(y2_D) + np.exp(y3_D)
print(sum_exp_ˆy_D)
cond_probs = calc_cond_probs(y1_D, y2_D, y3_D)
print(cond_probs)
