import pickle
import numpy as np


pickle_file_path_a = "output_results_a.pickle"
pickle_file_path_b = "output_results_b.pickle"

with open(pickle_file_path_a, 'rb') as f:
    matrix_a = pickle.load(f)

with open(pickle_file_path_b, 'rb') as f:
    matrix_b = pickle.load(f)


combined_matrix = np.vstack((matrix_a, matrix_b))
num_rows = combined_matrix.shape[0]

combined_matrix[3:, [0, 2]] = combined_matrix[3:, [2, 0]]
combined_matrix[3:, [1, 3]] = combined_matrix[3:, [3, 1]]


average_values = (combined_matrix[:, 1].astype(float) + combined_matrix[:, 3].astype(float)) / 2

combined_matrix = np.insert(combined_matrix, 4, average_values, axis=1)

sorted_indices = np.argsort(combined_matrix[:, 4])[::-1]
sorted_matrix = combined_matrix[sorted_indices]


for index, row in enumerate(sorted_matrix, start=1):
    url = row[0]
    url_similarity = row[1]
    review_text = row[2]
    review_similarity = row[3]
    composite_score = row[4]
    
    print(f"Order: {index}")
    print(f"URL: {url}")
    print(f"Similarity with Input URL: {url_similarity}")
    print(f"Review Text: {review_text}")
    print(f"Similarity with Input Review Text: {review_similarity}")
    print(f"Composite Score: {composite_score}\n")
