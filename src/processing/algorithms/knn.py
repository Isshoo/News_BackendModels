import numpy as np


class ManualKNN:
    def __init__(self, n_neighbors=7):
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)

    def predict(self, X_test):
        X_test = np.array(X_test)
        predictions = []

        for test_vector in X_test:
            # Hitung Euclidean Distance ke semua titik latih
            distances = np.linalg.norm(self.X_train - test_vector, axis=1)

            # Ambil indeks dari k tetangga terdekat
            nearest_indices = np.argsort(distances)[:self.n_neighbors]

            # Ambil label dari k tetangga
            nearest_labels = self.y_train[nearest_indices]

            # Voting untuk label terbanyak
            unique_labels, counts = np.unique(
                nearest_labels, return_counts=True)
            predicted_label = unique_labels[np.argmax(counts)]

            predictions.append(predicted_label)

        return np.array(predictions)
