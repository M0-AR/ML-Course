# ---
"""
02450ex_Fall2021_sol-1.pdf
Question 25
"""
import math

def time(K):
    to = 800 * math.log(800, 2) + 200 # Time for outer loop
    ti = 2400 * ((K - 1) * math.log(800 * (K - 1)/K, 2)) + 2400 # Time for inner loop
    return to + ti

K = 9
t_total = time(K)
while t_total <= 200000:
    K += 1
    t_total = time(K)

print(K-1)
# 9