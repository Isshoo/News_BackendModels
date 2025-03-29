import numpy as np
from collections import Counter


class CustomC5:
    def __init__(self):
        self.topic_data = {}
        self.entropy_overall = None

    def compute_overall_entropy(self, y_train):
        label_counts = Counter(y_train)
        total_samples = len(y_train)
        self.entropy_overall = -sum((count / total_samples) * np.log2(
            count / total_samples) for count in label_counts.values())
        return self.entropy_overall

    def fit(self, X_train, y_train):
        word_counts = {}
        for label in np.unique(y_train):
            indices = np.where(y_train == label)[0]
            word_freq = Counter()
            for i in indices:
                word_freq.update(X_train[i].split())
            total_words = sum(word_freq.values())
            entropy = -sum((count / total_words) * np.log2(count / total_words)
                           for count in word_freq.values())
            self.topic_data[label] = {
                'entropy': entropy, 'word_freq': word_freq}

    def compute_information_gain(self, text):
        gains = {}
        words = text.split()
        for label, data in self.topic_data.items():
            word_scores = [data['word_freq'].get(word, 0) for word in words]
            avg_score = np.mean(word_scores) if word_scores else 0
            gains[label] = avg_score

        sorted_gains = sorted(gains.items(), key=lambda x: x[1], reverse=True)

        if len(sorted_gains) > 1 and sorted_gains[0][1] == sorted_gains[1][1]:
            return None, sorted_gains[:2]  # Butuh KNN
        return sorted_gains[0][0], None  # Klasifikasi langsung
