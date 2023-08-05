"""
This code is performing clustering on a 1-dimensional dataset using the k-means algorithm, and you can understand the main components as follows:

Data Points (X): The array X contains 10 one-dimensional data points that you want to cluster.

Number of Clusters (nClusters): You have set the number of clusters you want to form to 3.

Initialization Option (withInitialCluster): A Boolean flag withInitialCluster is used to determine whether or not to use a specific initialization for the cluster centers.

If withInitialCluster is True, the initial cluster centers are defined by the initial_cluster array, which consists of the first three values of X.
If withInitialCluster is False, the k-means algorithm will be initialized with random cluster centers (with 100 different initializations).
K-means Clustering: Depending on the initialization, you create a k-means model with the specified parameters.

n_clusters specifies the number of clusters (3 in this case).
n_init specifies the number of times the algorithm will run with different centroid seeds if withInitialCluster is False.
max_iter sets the maximum number of iterations for a single run.
init sets the method for initialization. In the case of withInitialCluster=True, it's set to the initial_cluster.
Fitting the Model: The fit method is called on the k-means object to perform the clustering on the given data points in X.

Results:

Cluster centers are the coordinates of the centroids of the clusters. Labels are the cluster labels assigned to each
data point, indicating which cluster center is closest. Keep in mind that the provided code snippet will reshape the
initial_cluster array and X to be two-dimensional since scikit-learn expects the input data to be in this format,
even if it's one-dimensional.

The result will depend on the initialization and the nature of the algorithm. If initialized with specific centers,
the results will be consistent. If initialized randomly, the results might vary between different runs of the code. """
import numpy as np
from sklearn.cluster import KMeans


def perform_kmeans(X, n_clusters, with_initial_cluster=False, initial_cluster=None):
    """
    Performs KMeans clustering on the given data.

    Args:
        X (numpy.array): Data to cluster.
        n_clusters (int): The number of clusters to form.
        with_initial_cluster (bool, optional): Whether to use initial cluster centers. Defaults to False.
        initial_cluster (numpy.array, optional): Initial cluster centers. Required if with_initial_cluster is True.

    Returns:
        kmeans (KMeans): The fitted KMeans object.
    """
    if with_initial_cluster:
        if initial_cluster is None:
            raise ValueError("Initial cluster centers must be provided when with_initial_cluster is True.")
        kmeans = KMeans(n_clusters=n_clusters, n_init=1, max_iter=1000, init=initial_cluster).fit(X)
    else:
        kmeans = KMeans(n_clusters=n_clusters, n_init=100, max_iter=1000).fit(X)

    return kmeans


if __name__ == "__main__":
    # Example data
    X = np.array([-2.1, -1.7, -1.5, -0.4, 0.0, 0.6, 0.8, 1.0, 1, 1]).reshape(-1, 1)

    # Number of clusters
    nClusters = 3

    # Option to start with an initial cluster
    withInitialCluster = True
    initial_cluster = np.array([-2.1, -1.7, -1.5]).reshape(-1, 1)

    # Perform KMeans clustering
    kmeans_result = perform_kmeans(X, nClusters, withInitialCluster, initial_cluster)

    # Print the results
    print("Cluster centers = {}".format(kmeans_result.cluster_centers_))
    print("Labels = {}".format(kmeans_result.labels_))

"""
The output of the k-means clustering algorithm represents the final clustering result for the given data points. Let's break down what these specific results mean:

Cluster Centers:
Cluster Center 0: [-2.1] - This is the centroid for the first cluster.
Cluster Center 1: [-1.6] - This is the centroid for the second cluster.
Cluster Center 2: [0.57142857] - This is the centroid for the third cluster.
These centers represent the "average" or "middle" of the points within each cluster.
Labels: This array [0 1 1 2 2 2 2 2 2 2] represents the cluster assignment for each data point in X. Each value corresponds to the cluster label of the corresponding data point in X.

The first data point [-2.1] is assigned to cluster 0. The second and third data points [-1.7, -1.5] are assigned to 
cluster 1. The remaining data points [-0.4, 0.0, 0.6, 0.8, 1.0, 1,1] are assigned to cluster 2. The clustering result 
has partitioned the data points into three groups based on their values. Cluster 0 contains the lowest value, 
cluster 1 contains the next two lowest values, and cluster 2 contains all the remaining higher values. 

The specific values of the cluster centers and labels could vary if the algorithm was run with different 
initializations or parameters, but in this case, you've controlled the initialization, so the output should be 
consistent. """