# Tokenization, linear algebra, and easy handling of datasets
import spacy
import numpy as np
import pandas as pd

# Regular expression, string processing and efficient counting
import re
import string
from collections import Counter

from sklearn.model_selection import train_test_split

from Datasets.categorical_label_encoder import MultiColumnLabelEncoder



class Preprocess():
    def __init__(self):
        self.test_size = 0.2
        #self.seed = 200

    def encode_target_attribute(self, df):
        processed_df = MultiColumnLabelEncoder(columns = ['device_report_product_code']).fit_transform(df)
        return processed_df

    def count_words(self, processed_df):
        counts = Counter()
        for _, row in processed_df.iterrows():
            counts.update(row['tokenized_text'])
        return counts

    def delete_less_frequent_words(self, counts, frequency_threshold):
        print(f"\nOriginal number of words: {len(counts.keys())}")

        for word in list(counts):
            if counts[word] < frequency_threshold:
                del counts[word]

        print(f"After deleting less frequent words: {len(counts.keys())}")
        return counts
    
    def create_word_to_idx(self, counts):
        word_to_idx = {"": 0, "UNK": 1}
        words = ["", "UNK"]
        for word in counts:
            word_to_idx[word] = len(words)
            words.append(word)

        return word_to_idx, len(words)
    
    def encode_sentence(self, text, vocabulary, encoding_length):
        encoding_mask = np.zeros(encoding_length, dtype=int)
        to_be_encoded = np.array([vocabulary.get(word, vocabulary["UNK"]) for word in text])
        length = min(encoding_length, len(to_be_encoded))
        encoding_mask[:length] = to_be_encoded[:length]

        return encoding_mask

    def encode_texts(self, processed_df, vocabulary, encoding_length):
        processed_df['encoded text'] = processed_df['tokenized_text'].apply(lambda x: np.array(self.encode_sentence(x, vocabulary, encoding_length)))
        return processed_df

    def split_dataset(self, processed_df, seed):
        x = list(processed_df['encoded text'])
        y = list(processed_df['device_report_product_code'])
        x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=self.test_size, random_state=seed)
        return x_train, x_valid, y_train, y_valid