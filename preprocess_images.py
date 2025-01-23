import cv2
import numpy as np
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
import os


def download_image(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            return np.array(image)
        else:
            return None
    except Exception as e:
        print(f"Error downloading image from {url}: {e}")
        return None


def preprocess_image(image):
    if image is None:
        return None
    preprocessed_images = []
    
    resized_image = cv2.resize(image, (224, 224)) 
    preprocessed_images.append(resized_image)

    
    flip_horizontal = np.random.choice([True, False])
    if flip_horizontal:
        flipped_image = cv2.flip(resized_image, 1)  
        preprocessed_images.append(flipped_image)

    
    angles = [0, 45, 90, 135, 180, 225, 270, 315]
    for angle in angles:
        rotated_image = np.array(Image.fromarray(resized_image).rotate(angle))
        preprocessed_images.append(rotated_image)

    
    alpha_values = [1.0, 1.5, 2.0]  
    beta_values = [0, 30, 60]  
    for alpha in alpha_values:
        for beta in beta_values:
            adjusted_image = cv2.convertScaleAbs(resized_image, alpha=alpha, beta=beta)
            preprocessed_images.append(adjusted_image)

    return preprocessed_images


def save_images(images, output_directory):
    if images is None:
        return

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for i, img in enumerate(images):
        output_path = os.path.join(output_directory, f"preprocessed_image_{i}.jpg")
        cv2.imwrite(output_path, img)
        print(f"Pre-processed image saved as:", output_path)


def process_images_from_csv(csv_file_path, output_parent_directory):
    df = pd.read_csv(csv_file_path)
    for index, row in df.iterrows():
        image_links = row['Image'].strip('][').split(', ')  
        list_directory = os.path.join(output_parent_directory, f"list_{index + 1}")  
        images = []
        for link in image_links:
            image = download_image(link.strip("'"))
            preprocessed_images = preprocess_image(image)
            if preprocessed_images is not None:
                images.extend(preprocessed_images)
        save_images(images, list_directory)


csv_file_path = "C:/Users/ccbhi/Downloads/A2_Data.csv"
output_parent_directory = "preprocessed_images"
process_images_from_csv(csv_file_path, output_parent_directory)


