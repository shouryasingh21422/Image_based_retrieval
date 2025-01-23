import numpy as np
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle


def download_image(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            return image
        else:
            return None
    except Exception as e:
        print(f"Error downloading image from {url}: {e}")
        return None


def extract_features(image):
    if image is None:
        return None
    model = VGG16(weights='imagenet', include_top=False, pooling='avg')
    image = image.resize((224, 224))  
    image = np.expand_dims(image, axis=0)  
    image = preprocess_input(image)  
    features = model.predict(image)
    return features.flatten()


def normalize_features(features):
    if features is None:
        return None
    normalized_features = (features - features.mean()) / features.std()  
    return normalized_features


def calculate_cosine_similarity(input_features, target_features):
    input_features = input_features.reshape(1, -1)  
    target_features = target_features.reshape(1, -1)  
    similarity_scores = cosine_similarity(input_features, target_features)
    return similarity_scores


csv_file_path = "C:/Users/ccbhi/Downloads/A2_Data.csv"
df = pd.read_csv(csv_file_path)


input_image_url = input("Enter the URL of the image: ")

input_text = input("Enter the review text: ")

input_image = download_image(input_image_url)
if input_image is None:
    print("Failed to download the input image.")
else:
    input_features = extract_features(input_image)
    if input_features is None:
        print("Failed to extract features from the input image.")
    else:
        input_features = normalize_features(input_features)
        similarity_scores = []

        
        df['Review Text'].fillna('', inplace=True)
        tfidf_vectorizer = TfidfVectorizer()
        text_features = tfidf_vectorizer.fit_transform(df['Review Text'])
        input_text_features = tfidf_vectorizer.transform([input_text])
        
        
        text_similarity_scores = [cosine_similarity(input_text_features, text_features[i])[0][0] for i in range(len(df))]

        
        top_3_indices = np.argsort(text_similarity_scores)[::-1][:3]

        
        output_matrix = []

        
        for index in top_3_indices:
            review_text = df.loc[index, 'Review Text']
            review_similarity_score = text_similarity_scores[index]

           
            image_links = [link.strip().strip("[]").strip("'") for link in df.loc[index, 'Image'].split(',')]
            for link in image_links:
                image = download_image(link)
                if image is not None:
                    target_features = extract_features(image)
                    if target_features is not None:
                        target_features = normalize_features(target_features)
                        similarity_score = calculate_cosine_similarity(input_features, target_features)
                        output_matrix.append([review_text, review_similarity_score, link, similarity_score[0][0]])

                        
                        print("Review Text:", review_text)
                        print("Review Similarity Score:", review_similarity_score)
                        print("Image URL:", link)
                        print("Image Similarity Score:", similarity_score[0][0])
                        print()

                    else:
                        print("Failed to extract features from the image.")
                else:
                    print("Failed to download the image.")
        
        
        output_pickle_file_path = "output_results_b.pickle"

        
        with open(output_pickle_file_path, 'wb') as f:
            pickle.dump(output_matrix, f)

        print(f"Output matrix saved to {output_pickle_file_path} as a pickle file.")
