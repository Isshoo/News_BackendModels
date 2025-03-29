import numpy as np
from collections import Counter


class ManualKNN:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = np.array(y_train)

    def compute_euclidean_distance(self, vec1, vec2):
        return np.linalg.norm(vec1 - vec2)

    def predict(self, X_test_vector, relevant_labels):
        relevant_indices = [i for i, label in enumerate(
            self.y_train) if label in relevant_labels]
        X_train_filtered = self.X_train[relevant_indices]
        y_train_filtered = self.y_train[relevant_indices]

        distances = [self.compute_euclidean_distance(
            X_train_filtered[i], X_test_vector) for i in range(len(X_train_filtered))]
        nearest_indices = np.argsort(distances)[:self.n_neighbors]
        nearest_labels = y_train_filtered[nearest_indices]

        return Counter(nearest_labels).most_common(1)[0][0]
