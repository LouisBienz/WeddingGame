import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Define 5 vectors of dimension 7
vectors = np.array([
    [1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 2],
    [1, 1, 1, 1, 1, 1, 3],
    [1, 1, 1, 1, 1, 1, 4],
    [9, 9, 9, 9, 9, 9, 9]
])

# Function to find the vector with the largest angular distance
def find_most_dissimilar(vectors, max_selections=1):
    sim_matrix = cosine_similarity(vectors)
    np.fill_diagonal(sim_matrix, 2)  # Ignore self-similarity
    dissimilar_indices = []
    selection_count = np.zeros(len(vectors))

    for i in range(len(vectors)):
        dissimilarities = sim_matrix[i]
        valid_indices = np.where(selection_count < max_selections)[0]
        valid_indices = valid_indices[valid_indices != i]  # Exclude self
        if len(valid_indices) == 0:
            # If all vectors have reached their max selections, reset the counts
            selection_count[:] = 0
            valid_indices = np.arange(len(vectors))
            valid_indices = valid_indices[valid_indices != i]

        most_dissimilar = valid_indices[np.argmin(dissimilarities[valid_indices])]
        dissimilar_indices.append(most_dissimilar)
        selection_count[most_dissimilar] += 1

    return dissimilar_indices

# Find the most dissimilar vectors
most_dissimilar_indices = find_most_dissimilar(vectors)

# Print results
for i in range(len(vectors)):
    print(f"Vector {i}: Most dissimilar is Vector {most_dissimilar_indices[i]}")
