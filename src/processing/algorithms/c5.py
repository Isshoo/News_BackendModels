import numpy as np
from collections import Counter


class CustomC5:
    def __init__(self):
        self.topic_data = {}
        self.entropy_overall = None

    def compute_entropy(self, labels):
        label_counts = Counter(labels)
        total_samples = len(labels)
        probs = np.array(list(label_counts.values())) / total_samples
        return -np.sum(probs * np.log2(probs)) if total_samples > 0 else 0

    def compute_word_entropy(self, word, dataset, labels):
        word_occurrences = [labels[i]
                            for i, text in enumerate(dataset) if word in text.split()]
        return self.compute_entropy(word_occurrences)

    def compute_entropy_without_word(self, word, dataset, labels):
        filtered_labels = [labels[i] for i, text in enumerate(
            dataset) if word not in text.split()]
        return self.compute_entropy(filtered_labels)

    def compute_information_gain(self, S, word, dataset, labels):
        H_S = self.compute_entropy(labels)
        H_word = self.compute_word_entropy(word, dataset, labels)
        H_without_word = self.compute_entropy_without_word(
            word, dataset, labels)

        S_word = sum(1 for text in dataset if word in text.split())
        S_not_word = len(dataset) - S_word

        IG = H_S - ((S_word / len(dataset)) * H_word +
                    (S_not_word / len(dataset)) * H_without_word)
        return IG

    def fit(self, X_train, y_train):
        unique_labels, label_indices = np.unique(y_train, return_inverse=True)
        word_counts = {label: Counter() for label in unique_labels}

        for text, label_idx in zip(X_train, label_indices):
            word_counts[unique_labels[label_idx]].update(text.split())

        self.topic_data = {
            label: {
                'entropy': self.compute_entropy(list(word_freq.elements())),
                'word_freq': word_freq
            }
            for label, word_freq in word_counts.items()
        }

    def predict(self, text):
        words = text.split()
        gains = {}

        for label, data in self.topic_data.items():
            word_scores = [data['word_freq'].get(word, 0) for word in words]
            avg_score = np.mean(word_scores) if word_scores else 0
            gains[label] = avg_score

        sorted_gains = sorted(gains.items(), key=lambda x: x[1], reverse=True)

        if len(sorted_gains) > 1 and sorted_gains[0][1] == sorted_gains[1][1]:
            return None, sorted_gains[:2]  # Butuh KNN
        return sorted_gains[0][0], None  # Klasifikasi langsung
