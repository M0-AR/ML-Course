import numpy as np

"""NearestNeighbors class: This class encapsulates the operations related to nearest neighbors, density, 
and ARD calculation. It requires K to be initialized, which is the number of nearest neighbors. 

density method: This method calculates the density given distances to the nearest neighbors. It also checks if K 
matches the number of nearest neighbors provided. 

ARD method: This method calculates the ARD given the density of the observation and the density of the nearest other 
observations. It also checks if K matches the number of density values provided for other observations. 

Error handling: By including appropriate error handling, the code becomes more robust and helps diagnose potential 
issues with mismatched sizes. 

Main code block: In the main part of the code, the NearestNeighbors class is instantiated with K=2, and the density 
and ARD are calculated for the given data. The results are printed to the console. """


class NearestNeighbors:
    def __init__(self, K):
        self.K = K

    def density(self, distanceNearestNeighbors):
        if self.K != np.size(distanceNearestNeighbors):
            raise ValueError("K does not match the number of nearest neighbors provided")

        density_value = 1 / (1 / self.K * np.sum(distanceNearestNeighbors))
        return density_value

    def ard(self, densityObservation, densityNearestOtherObservations):
        if self.K != np.size(densityNearestOtherObservations):
            raise ValueError("K does not match the number of density values provided for other observations")

        ard_value = densityObservation / (1 / self.K * np.sum(densityNearestOtherObservations))
        return ard_value


if __name__ == "__main__":
    K = 2
    nn = NearestNeighbors(K)

    # Distances to nearest neighbors for the desired point (observation 1)
    distNearestNeighbors1 = [1.04, 1.88]
    density1 = nn.density(distNearestNeighbors1)
    print("Density for observation: ", density1)

    # Distances to nearest neighbors for the K points closest to our point...
    distOthers = [
        [0.63, 1.02],  # K nearest neighbors distance for observation 2
        [1.04, 1.80]  # K nearest neighbors distance for observation 3
    ]

    densityOthers = [nn.density(dist) for dist in distOthers]
    ard1 = nn.ard(density1, densityOthers)
    print("ARD for observation: ", ard1)
