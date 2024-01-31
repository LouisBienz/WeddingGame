import numpy as np
from typing import List
from scipy.spatial.distance import pdist, squareform

vectors = np.array([
    [1, 1, 1, 1],
    [2, 2, 2, 2],
    [3, 3, 3, 3],
    [4, 4, 4, 4],
    [5, 5, 5, 5],
    [6, 6, 6, 6],
    [7, 7, 7, 7],
    [8, 8, 8, 8],
    [9, 9, 9, 9]
])

def calculate_euclidean_distances(vectors: np.ndarray) -> np.ndarray:
    """
    Calculate the pairwise Euclidean distances between vectors.

    Args:
    vectors (np.ndarray): An array of vectors.

    Returns:
    np.ndarray: A 2D array representing the pairwise Euclidean distances.
    """
    return squareform(pdist(vectors, 'euclidean'))

def modify_distances(distances: np.ndarray) -> np.ndarray:
    """
    Modify the distances using a custom function.

    Args:
    distances (np.ndarray): The original distance matrix.

    Returns:
    np.ndarray: The modified distance matrix.
    """


    # Idea: Riemann sphere. In detail:
    # Given r, define cr = r + i0, where i is the imaginary unit
    # Find the intersection between 
    #   - the line segment (1i) to cr
    #   - the circle with radius 0.5 and center (0+0.5 i)
    # Pick finally the real part of the complex intersection point.
    def custom_function(r):
        return 1 / (1 + r**2)
    
    # Apply the custom function to each element in the distance matrix
    return np.vectorize(custom_function)(distances) 


def nearest_neighbor_algorithm(distance_matrix: np.ndarray) -> List[int]:
    """
    Solve the Traveling Salesman Problem using the Nearest Neighbor algorithm.

    Args:
    distance_matrix (np.ndarray): A 2D array representing the distances between each pair of points.

    Returns:
    List[int]: The order of nodes visited in the TSP solution.
    """
    n = len(distance_matrix)
    visited = [False] * n
    path = [0]
    visited[0] = True

    for _ in range(1, n):
        last = path[-1]
        next_node = None
        min_dist = np.inf

        for i in range(n):
            if not visited[i] and distance_matrix[last][i] < min_dist:
                min_dist = distance_matrix[last][i]
                next_node = i

        visited[next_node] = True
        path.append(next_node)

    return path

distances = calculate_euclidean_distances(vectors)

modified_distances = modify_distances(distances)

# Solve TSP with modified distances
tsp_path_opposite = nearest_neighbor_algorithm(modified_distances)
tsp_path_similar = nearest_neighbor_algorithm(distances)

# Print the path
print("Modified TSP Path:", tsp_path_opposite)
print("Modified TSP Path:", tsp_path_similar)