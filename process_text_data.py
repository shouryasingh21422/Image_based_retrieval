import os
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
import string
from collections import Counter
import math
import pickle


def preprocess_text(text):
    
    text = str(text)
    
    
    text = text.lower()
    
    
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    
    
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens if token not in stop_words and token not in string.punctuation]
    
    
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return tokens


def calculate_tf_idf(documents):
    
    tf = [{token: count / len(document) for token, count in Counter(document).items()} for document in documents]
    
    
    df = Counter()
    for document in documents:
        df.update(set(document))
    
    
    idf = {token: math.log(len(documents) / df[token]) for token in df}
    
    
    tf_idf = [{token: tf_value * idf[token] for token, tf_value in doc.items()} for doc in tf]
    
    return tf_idf


def save_tf_idf(tf_idf_scores, output_dir):
    
    for i, scores in enumerate(tf_idf_scores):
        txt_file_path = os.path.join(output_dir, f'tfidf_scores_{i+1}.txt')
        pickle_file_path = os.path.join(output_dir, f'tfidf_scores_{i+1}.pkl')
        
        with open(txt_file_path, 'w') as txt_file:
            with open(pickle_file_path, 'wb') as pickle_file:
                pickle.dump(scores, pickle_file)
                for token, score in scores.items():
                    txt_file.write(f'{token}: {score}\n')
    
    print("TF-IDF scores calculated and saved in both text and pickle formats in the specified directory.")


def process_data(csv_path, output_dir):
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    
    df = pd.read_csv(csv_path)

    
    if 'Review Text' not in df.columns:
        raise ValueError("Column 'Review Text' not found in the CSV file.")

    
    documents = [preprocess_text(review) for review in df['Review Text']]

    
    tf_idf_scores = calculate_tf_idf(documents)

    
    save_tf_idf(tf_idf_scores, output_dir)


csv_path = "C:/Users/ccbhi/Downloads/A2_Data.csv"
output_dir = "C:/IIITD/sem6/IR/A2/tfidf_scores"


process_data(csv_path, output_dir)
