import numpy as np
from scipy.spatial.distance import euclidean

# Define 5 vectors of dimension 7
vectors = np.array([
    [1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1],
    [9, 9, 9, 9, 9, 9, 9]
])

# Function to find the vector with the smallest Euclidean distance
def find_most_similar(vectors, max_selections=1):
    similar_indices = []
    selection_count = np.zeros(len(vectors))

    for i, v in enumerate(vectors):
        # Calculate distances excluding the vector itself
        distances = np.array([euclidean(v, vectors[j]) for j in range(len(vectors)) if j != i])
        # Valid indices excluding the vector itself
        valid_indices = np.array([j for j in range(len(vectors)) if j != i and selection_count[j] < max_selections])

        if len(valid_indices) == 0:
            # Reset the counts if all vectors have reached their max selections
            selection_count[:] = 0
            valid_indices = np.array([j for j in range(len(vectors)) if j != i])

        # Adjust indices for the distances array
        adjusted_indices = np.array([np.where(valid_indices == j)[0][0] for j in valid_indices])

        # Find the most similar among the valid indices
        most_similar = valid_indices[np.argmin(distances[adjusted_indices])]
        similar_indices.append(most_similar)
        selection_count[most_similar] += 1

    return similar_indices

# Function to find the vector with the largest Euclidean distance
def find_most_dissimilar(vectors, max_selections=1):
    dissimilar_indices = []
    selection_count = np.zeros(len(vectors))

    for i, v in enumerate(vectors):
        # Calculate distances excluding the vector itself
        distances = np.array([euclidean(v, vectors[j]) for j in range(len(vectors)) if j != i])
        # Valid indices excluding the vector itself
        valid_indices = np.array([j for j in range(len(vectors)) if j != i and selection_count[j] < max_selections])
        
        if len(valid_indices) == 0:
            # Reset the counts if all vectors have reached their max selections
            selection_count[:] = 0
            valid_indices = np.array([j for j in range(len(vectors)) if j != i])

        # Adjust indices for the distances array
        adjusted_indices = np.array([np.where(valid_indices == j)[0][0] for j in valid_indices])
        
        # Find the most dissimilar among the valid indices
        most_dissimilar = valid_indices[np.argmax(distances[adjusted_indices])]
        dissimilar_indices.append(most_dissimilar)
        selection_count[most_dissimilar] += 1

    return dissimilar_indices

# Find the most similar vectors
most_similar_indices = find_most_similar(vectors)

# Find the most dissimilar vectors
most_dissimilar_indices = find_most_dissimilar(vectors)

# Print results
for i in range(len(vectors)):
    print(f"Vector {i}: Most similar is Vector {most_similar_indices[i]}")
    print(f"Vector {i}: Most dissimilar is Vector {most_dissimilar_indices[i]}")

    






