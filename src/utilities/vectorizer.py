import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


class TextVectorizer:
    vectorizer = TfidfVectorizer()

    def __init__(self):
        # Initialize your vectorizer (like TfidfVectorizer)
        pass

    def fit_transform(self, corpus):
        """Transform the corpus into a matrix of TF-IDF features"""
        # Ensure corpus is a pandas Series before checking for emptiness
        if isinstance(corpus, pd.Series):
            if corpus.empty:
                raise ValueError("Corpus is empty. Please provide valid data.")
        else:
            raise TypeError("Expected input corpus to be a pandas Series.")

        # Perform vectorization (you can add additional checks or handling here)
        return self.vectorizer.fit_transform(corpus)

    def transform(self, corpus):
        """Transform the corpus into a matrix of TF-IDF features"""
        if isinstance(corpus, pd.Series):
            if corpus.empty:
                raise ValueError("Corpus is empty. Please provide valid data.")
        else:
            raise TypeError("Expected input corpus to be a pandas Series.")

        return self.vectorizer.transform(corpus)

    def save_vectorizer(self, path):
        """Save the trained vectorizer to a file"""
        with open(path, 'wb') as f:
            pickle.dump(self.vectorizer, f)

    def load_vectorizer(self, path):
        """Load a previously saved vectorizer"""
        with open(path, 'rb') as f:
            self.vectorizer = pickle.load(f)
