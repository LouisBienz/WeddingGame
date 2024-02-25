import numpy as np
from typing import List
from scipy.spatial.distance import pdist, squareform
from Guest import Guest
from handle_csv import import_csv

def calculate_euclidean_distances(data_entries: List[Guest]) -> np.ndarray:
    """
    Calculate the pairwise Euclidean distances between data entries.

    Args:
    data_entries (List[Guest]): A list of Guest objects.

    Returns:
    np.ndarray: A 2D array representing the pairwise Euclidean distances.
    """
    vectors = np.array([entry.answers for entry in data_entries])
    return squareform(pdist(vectors, 'euclidean'))

def modify_distances(distances: np.ndarray) -> np.ndarray:
    """
    Modify the distances using a custom function.

    Args:
    distances (np.ndarray): The original distance matrix.

    Returns:
    np.ndarray: The modified distance matrix.
    """
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

if __name__ == "__main__":
    print("Import CSV")
    headers, data = import_csv()
    print("Successful")
    
    print("-----------------")

    print("Converting CSV to Guest objects")
    guests = [Guest(entry) for entry in data]
    print("First Guest: ", guests[0].name)
    print("Successful")
    
    print("-----------------")

    distances = calculate_euclidean_distances(guests)
    modified_distances = modify_distances(distances)

    # Solve TSP with modified distances
    tsp_path_opposite = nearest_neighbor_algorithm(modified_distances)
    tsp_path_similar = nearest_neighbor_algorithm(distances)

    guest_names = [entry.name for entry in guests]

    # Print the TSP path with guest names
    print("Modified TSP Path:")
    for index in tsp_path_opposite:
        print(guest_names[index])

    print("-----------------")

    print("Original TSP Path:")
    for index in tsp_path_similar:
        print(guest_names[index])

    # Print the path
    print("Modified TSP Path:", tsp_path_opposite)
    print("Original TSP Path:", tsp_path_similar)
