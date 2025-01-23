import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle


csv_file_path = "C:/Users/ccbhi/Downloads/A2_Data.csv"
df = pd.read_csv(csv_file_path)
df['Image'] = df['Image'].apply(lambda x: x.strip("[]").replace("'", "").split(', '))
df = df.explode('Image')


normalized_features_dict_file = "normalized_features/normalized_features_dict.pickle"
with open(normalized_features_dict_file, 'rb') as f:
    normalized_features_dict = pickle.load(f)


input_image_url = input("Enter the URL of the image: ")
input_text = input("Enter the review text: ")


if input_image_url in normalized_features_dict:
    input_features = normalized_features_dict[input_image_url]
else:
    print("Input URL not found in the precomputed features dictionary.")
    exit()

print("\nComparing input URL with the following URLs from the CSV:")

similarity_scores = []


input_image_url_standardized = input_image_url.strip("[]").replace("'", "")


for url, target_features in normalized_features_dict.items():
    
    url_standardized = url.strip("[]").replace("'", "")
    
    
    similarity_score = cosine_similarity([input_features], [target_features])[0][0]
    similarity_scores.append((url_standardized, similarity_score))


similarity_scores.sort(key=lambda x: x[1], reverse=True)


print("\nTop 3 similar URLs:")
for i in range(min(3, len(similarity_scores))):
    url, score = similarity_scores[i]
    print(f"URL: {url}, Similarity Score: {score}")


top_3_urls = [url for url, _ in similarity_scores[:3]]
sk=[i for j, i in similarity_scores[:3]]


top_3_review_text = []
for url in top_3_urls:
    matching_rows = df[df['Image'] == url]
    if not matching_rows.empty:
        top_3_review_text.append(matching_rows.iloc[0]['Review Text'])
    else:
        top_3_review_text.append("No review text found")


df['Review Text'].fillna('', inplace=True)
tfidf_vectorizer = TfidfVectorizer()
text_features = tfidf_vectorizer.fit_transform(df['Review Text'])

input_text_features = tfidf_vectorizer.transform([input_text])


top_3_text_features = tfidf_vectorizer.transform(top_3_review_text)
text_similarity_scores = cosine_similarity(input_text_features, top_3_text_features)[0]


print("\nCorresponding review texts and similarity scores:")
for text, score in zip(top_3_review_text, text_similarity_scores):
    print(f"Review Text: {text}, Similarity Score: {score}")


output_file_path = "C:/IIITD/sem6/IR/A2/q3_a_results.txt"


with open(output_file_path, 'w') as f:
    
    for url, text, score in zip(top_3_urls, top_3_review_text, text_similarity_scores):
        
        f.write(f"URL: {url}, Review Text: {text}, Similarity Score: {score}\n")


print(f"\nTop 3 image-text pairs saved to {output_file_path}")


output_matrix = np.zeros((3, 4), dtype=object)
output_matrix[:, 0] = top_3_urls
output_matrix[:, 1] = sk
output_matrix[:, 2] = top_3_review_text
output_matrix[:, 3] = text_similarity_scores


output_pickle_file_path = "output_results_a.pickle"

with open(output_pickle_file_path, 'wb') as f:
    pickle.dump(output_matrix, f)

print(f"\nOutput saved to {output_pickle_file_path} as a pickle file.")