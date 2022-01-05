import re
import pandas as pd
import nltk
from tqdm import tqdm
nltk.download('stopwords')
from nltk.corpus import stopwords
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def preprocess_list(doc_list):
    # stopword list
    stop_words = stopwords.words("english")
    new_doc_list = []

    for doc in tqdm(doc_list, desc="Preprocessing"):
        doc = preprocess_doc(doc, stop_words)
        new_doc_list.append(doc)

    return new_doc_list
# end preprocess_list

def preprocess_doc(doc, stop_words):
    # 1. remove all numeric references of form [XX]
    doc = re.sub('[\[].[0-9]*[\]]', '', doc)
    doc = re.sub('[\(].*?[\)]', '', doc)
    # 2. remove newlines and multiple whitespace, lower case everything
    doc = re.sub('\s+', ' ', doc).strip()
    doc = doc.lower()

    # 3. remove special characters
    # Regex to keep . , and ' is [^A-Za-z0-9.,\' ]
    doc = re.sub('[^A-Za-z0-9 ]', '', doc)

    # 4. remove stopwords
    doc = " ".join([w for w in doc.split() if w not in stop_words])

    return doc
