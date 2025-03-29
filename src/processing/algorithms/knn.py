import numpy as np
from collections import Counter


class CustomKNN:
    def __init__(self, n_neighbors=5):
        self.n = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Menyimpan data latih KNN"""
        self.X_train = X_train  # Konversi ke array numpy
        self.y_train = np.array(y_train)

    def euclidean_distance(self, vec1, vec2):
        """Menghitung jarak Euclidean antara dua vektor"""
        return np.sqrt(np.sum((vec1 - vec2) ** 2))

    def predict(self, text_vector, candidate_labels):
        """Klasifikasi dengan KNN menggunakan kandidat dari C5.0"""
        if self.X_train is None or self.y_train is None:
            raise ValueError(
                "Model belum dilatih. Jalankan `fit()` terlebih dahulu.")

        distances = [
            (i, self.euclidean_distance(text_vector, sample))
            for i, sample in enumerate(self.X_train)
        ]

        distances.sort(key=lambda x: x[1])  # Urutkan berdasarkan jarak
        nearest_labels = [self.y_train[i] for i, _ in distances[:self.n]]

        # Pastikan hanya memilih label yang ada di kandidat
        filtered_labels = [
            label for label in nearest_labels if label in candidate_labels]

        # Jika tidak ada kandidat yang cocok, pakai label mayoritas dari k tetangga
        return Counter(filtered_labels).most_common(1)[0][0] if filtered_labels else Counter(nearest_labels).most_common(1)[0][0]
