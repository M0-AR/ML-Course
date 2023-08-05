"""The code demonstrates two methods to project a data point onto given principal component vectors. This kind of
projection is commonly used in PCA for dimensionality reduction, where the original data is represented in a new
space defined by the principal components. By projecting the data onto these components, you can capture most of the
variability in the data using fewer dimensions, which can be useful for visualization or further analysis. """

"""The project_to_principal_components function takes a point x and a matrix V, whose rows are the principal 
component vectors, and returns the projections of the point onto the principal components. The print_projections 
function takes an array of projections and prints them to the console. The principal component vectors and the point 
to be projected are defined in the if __name__ == "__main__": block, and the functions are called to compute and 
print the projections. """

import numpy as np


def project_to_principal_components(x, V):
    """
    Projects a given point onto specified principal components.

    Parameters:
        x (array): The point to be projected.
        V (array): Matrix whose rows are the principal component vectors.

    Returns:
        array: Projections of the point onto the principal components.
    """

    # Compute the projection of the point onto each principal component
    b = np.matmul(V, x)

    return b


def print_projections(b):
    """
    Prints the projections to the console.

    Parameters:
        b (array): Projections of the point onto the principal components.
    """

    for i, projection in enumerate(b):
        print(f"Projection to PC{i + 1}: {projection}")


if __name__ == "__main__":
    # Principal component vectors
    V1 = np.array([0.45, -0.4, 0.58, 0.55])
    V2 = np.array([0.6, 0.2, -0.08, -0.3])

    # New point to project
    x = np.array([-1, -1, -1, 1])

    # Matrix whose rows are the principal component vectors
    V = np.array([V1, V2])

    # Projection on PC
    b = project_to_principal_components(x, V)

    print_projections(b)

"""The projections you obtained represent how much the data point x aligns with the directions defined by the 
principal component vectors V1 and V2. 

Here's what the projections mean in the context of Principal Component Analysis (PCA):

Projection to PC1 (-0.08): This value represents the coordinate of the data point x along the direction defined by 
the first principal component V1. Since the projection value is close to zero, this suggests that the data point is 
nearly orthogonal to V1, meaning it does not vary much along the direction defined by this principal component. 

Projection to PC2 (-1.02): This value represents the coordinate of the data point x along the direction defined by 
the second principal component V2. The value of -1.02 indicates that the data point is aligned with V2 but in the 
opposite direction, as indicated by the negative sign. 

Interpretation The projections provide a new representation of the data point x in the coordinate system defined by 
the principal components V1 and V2. This new representation can provide insights into how the data point relates to 
the main patterns of variation in your dataset. 

If V1 and V2 are the principal components corresponding to the largest singular values (i.e., they explain the most 
variance), then the projections reveal how the data point x aligns with the most significant directions of variation 
in the data. 

The projection values can also be used to plot the data point in a lower-dimensional space defined by the principal 
components, which can be useful for visualization or further analysis, such as clustering or classification. 

Keep in mind that the interpretation depends on the context and the specific data from which the principal components 
were derived. Understanding what the original features represent and how the principal components were computed would 
provide deeper insights into the meaning of these projections. """