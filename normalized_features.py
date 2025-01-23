import numpy as np
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
import os
import pickle
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input


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


def process_image_links(image_links):
    normalized_features_dict = {}
    if isinstance(image_links, str): 
        image_links = image_links.strip('][').split(', ') 
        for link in image_links:
            image = download_image(link.strip("'"))
            if image is not None:
                image_features = extract_features(image)
                if image_features is not None:
                    normalized_features = normalize_features(image_features)
                    normalized_features_dict[link.strip("'")] = normalized_features
    return normalized_features_dict


def save_features_to_pickle(features_dict, output_file):
    with open(output_file, 'wb') as f:
        pickle.dump(features_dict, f)


def process_csv_and_save_features(csv_file_path, output_parent_directory):
    df = pd.read_csv(csv_file_path)
    for index, row in df.iterrows():
        image_links = row['Image']
        normalized_features_dict = process_image_links(image_links)
        output_file = os.path.join(output_parent_directory, f"normalized_features_{index}.pickle")
        save_features_to_pickle(normalized_features_dict, output_file)



csv_file_path = "C:/Users/ccbhi/Downloads/A2_Data.csv"
output_parent_directory = "normalized_features"
process_csv_and_save_features(csv_file_path, output_parent_directory)


