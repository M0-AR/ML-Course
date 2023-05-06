######################################
#  the p-norm distance
import numpy as np

# Define the vectors
x14 = np.array([26, 0, 2, 0, 0, 0, 0])
x18 = np.array([19, 0, 0, 0, 0, 0, 0])

# Calculate the p-norm distances
d_inf = np.max(np.abs(x14 - x18))
d_1 = np.power(np.sum(np.power(np.abs(x14 - x18), 1)), 1 / 1)
d_3 = np.power(np.sum(np.power(np.abs(x14 - x18), 3)), 1 / 3)
d_4 = np.power(np.sum(np.power(np.abs(x14 - x18), 4)), 1 / 4)

# Print the distances
print("d_inf(x14, x18) = ", d_inf)
print("d_1(x14, x18) = ", d_1)
print("d_3(x14, x18) = ", d_3)
print("d_4(x14, x18) = ", d_4)
########################################

a1 = 0.73 * 0.274 + 0.81 * 0.23 + 0.70 * 0.244 + 0.68 * 0.252
print(a1)

a2 = 0.11 * 0.274 + 0.03 * 0.23 + 0.03 * 0.244 + 0.09 * 0.252
print(a2)
# print(0.84 * 0.23 + 0.03 * 0.23 + 0.1 * 0.23 + 0.06 * 0.23)
p = (0.84 * 0.23) / ((0.84 * 0.23 + 0.03 * 0.23 + 0.1 * 0.23 + 0.06 * 0.23)+a1)
print(p)
p = (0.84 * 0.23) / ((0.84 * 0.23 + 0.03 * 0.23 + 0.1 * 0.23 + 0.06 * 0.23)+a1+a2)
print(p)

p = (0.84 * 0.23) / ((0.84 * 0.23 + 0.03 * (1 - 0.23) + 0.1 * (1 - 0.23) + 0.06 * 0.23) + a1)
print(p)
p = (0.84 * 0.23) / ((0.84 * 0.23 + 0.03 * (1 - 0.23) + 0.1 * (1 - 0.23) + 0.06 * 0.23) + a1 + a2)
print(p)

p = (0.84 * 0.23) / ((0.84 * 0.23 + 0.03 * (1 - 0.23) + 0.1 * 0.23 + 0.06 * (1 - 0.23)) + a1)
print(p)
p = (0.84 * 0.23) / ((0.84 * 0.23 + 0.03 * (1 - 0.23) + 0.1 * 0.23 + 0.06 * (1 - 0.23)) + a1 + a2)
print(p)

########################################
import statistics

x = [0, 1, 1, 1, 2, 3, 4, 4, 5, 14]

mean = statistics.mean(x)
median = statistics.median(x)
mode = statistics.mode(x)

y = mean + median + mode

print(y)



import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# Define the pairwise cityblock distances between the 8 observations
D = np.array([[0, 4, 7, 9, 5, 5, 5, 6],
              [4, 0, 7, 7, 7, 3, 7, 8],
              [7, 7, 0, 10, 6, 6, 4, 9],
              [9, 7, 10, 0, 8, 6, 10, 9],
              [5, 7, 6, 8, 0, 8, 6, 7],
              [5, 3, 6, 6, 8, 0, 8, 11],
              [5, 7, 4, 10, 6, 8, 0, 7],
              [6, 8, 9, 9, 7, 11, 7, 0]])

# Apply hierarchical clustering with single linkage
Z = linkage(D, method='single')

# Define the labels for the x-axis
labels = ['o1', 'o2', 'o3', 'o4', 'o5', 'o6', 'o7', 'o8']

# Plot the resulting dendrogram
fig, ax = plt.subplots(figsize=(8, 5))
dendrogram(Z, labels=labels)
plt.show()
