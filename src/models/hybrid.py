import numpy as np
import pandas as pd
from src.utilities.vectorizer import TextVectorizer
from src.processing.algorithms.knn import CustomKNN
from src.processing.algorithms.c5 import CustomC5


class HybridClassifier:
    def __init__(self, n_neighbors=5):
        self.vectorizer = TextVectorizer()
        self.c5 = CustomC5()
        self.knn = CustomKNN(n_neighbors)
        self.vectorizer_path = './src/storage/vectorizers/vectorizer.pkl'
        # Untuk mengecek apakah vectorizer sudah di-train
        self.is_vectorizer_trained = False

    def fit(self, X_train, y_train):
        """Melatih C5.0 dan KNN dengan TF-IDF"""
        X_train_series = pd.Series(
            X_train)  # Convert to pandas Series if it's not already
        X_train_vectors = self.vectorizer.fit_transform(
            X_train_series).toarray()

        self.vectorizer.save_vectorizer(self.vectorizer_path)
        self.is_vectorizer_trained = True  # Set flag bahwa vectorizer sudah dilatih

        self.c5.fit(X_train, y_train)
        self.knn.fit(X_train_vectors, y_train)

    def predict(self, X_test):
        """Memprediksi kategori berdasarkan model Hybrid C5.0-KNN"""
        if not self.is_vectorizer_trained:
            self.vectorizer.load_vectorizer(self.vectorizer_path)

        X_test_vectors = self.vectorizer.transform(X_test).toarray()
        predictions = []

        for i, text in enumerate(X_test):
            label, candidates = self.c5.predict(text)  # C5 prediksi awal
            if label is not None:
                predictions.append(label)
            else:
                # Jika C5 tidak bisa memutuskan, gunakan KNN
                if candidates:
                    candidate_labels = [c[0] for c in candidates]
                else:
                    candidate_labels = None  # Gunakan semua label jika tidak ada kandidat

                knn_prediction = self.knn.predict(
                    X_test_vectors[i].reshape(1, -1), candidate_labels)
                predictions.append(knn_prediction)

        return predictions
