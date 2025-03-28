import numpy as np
from collections import Counter


class ManualKNN:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X_train, y_train, vectorizer):
        self.X_train = vectorizer.fit_transform(X_train).toarray()
        self.y_train = np.array(y_train)
        self.vectorizer = vectorizer

    def predict(self, X_test, relevant_labels):
        X_test_vector = self.vectorizer.transform([X_test]).toarray()
        relevant_indices = [i for i, label in enumerate(
            self.y_train) if label in relevant_labels]
        X_train_filtered = self.X_train[relevant_indices]
        y_train_filtered = self.y_train[relevant_indices]

        distances = np.linalg.norm(X_train_filtered - X_test_vector, axis=1)
        nearest_indices = np.argsort(distances)[:self.n_neighbors]
        nearest_labels = y_train_filtered[nearest_indices]

        return Counter(nearest_labels).most_common(1)[0][0]
