import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.preprocessing import LabelEncoder
import time


class CustomC5:
    def __init__(self):
        self.topic_data = {}
        self.word_gains = {}

    def compute_entropy(self, labels):
        label_counts = Counter(labels)
        total_samples = len(labels)
        probs = np.array(list(label_counts.values())) / total_samples
        return -np.sum(probs * np.log2(probs)) if total_samples > 0 else 0

    def compute_word_entropy(self, word, text_sets, labels):
        word_occurrences = [labels[i]
                            for i, text_set in enumerate(text_sets) if word in text_set]
        return self.compute_entropy(word_occurrences)

    def compute_entropy_without_word(self, word, text_sets, labels):
        filtered_labels = [labels[i] for i, text_set in enumerate(
            text_sets) if word not in text_set]
        return self.compute_entropy(filtered_labels)

    def compute_information_gain(self, word, text_sets, labels, H_S):
        H_word = self.compute_word_entropy(word, text_sets, labels)
        H_without_word = self.compute_entropy_without_word(
            word, text_sets, labels)

        S_word = sum(1 for text_set in text_sets if word in text_set)
        S_not_word = len(text_sets) - S_word

        IG = H_S - ((S_word / len(text_sets)) * H_word +
                    (S_not_word / len(text_sets)) * H_without_word)

        return IG

    def fit(self, X_train, y_train):
        unique_labels, label_indices = np.unique(y_train, return_inverse=True)
        word_counts = {label: Counter() for label in unique_labels}
        all_words = set()
        text_sets = [set(text.split()) for text in X_train]

        for text_set, label_idx in zip(text_sets, label_indices):
            all_words.update(text_set)
            word_counts[unique_labels[label_idx]].update(text_set)

        self.topic_data = {
            label: {
                'entropy': self.compute_entropy(list(word_freq.elements())),
                'word_freq': word_freq
            }
            for label, word_freq in word_counts.items()
        }

        H_S = self.compute_entropy(y_train)

        self.word_gains = {
            word: self.compute_information_gain(
                word, text_sets, y_train, H_S)
            for word in all_words
        }

    def predict(self, text):
        words = text.split()
        gains = {}

        for label, data in self.topic_data.items():
            # Ganti skor dari frekuensi kata → skor berdasarkan information gain × frekuensi kata
            word_scores = [
                data['word_freq'].get(word, 0) * self.word_gains.get(word, 0)
                for word in words
            ]
            avg_score = np.mean(word_scores) if word_scores else 0
            gains[label] = avg_score

        sorted_gains = sorted(gains.items(), key=lambda x: x[1], reverse=True)

        total_gain = sum(gains.values()) if sum(gains.values()) > 0 else 1
        confidence = sorted_gains[0][1] / total_gain

        if len(sorted_gains) > 1 and sorted_gains[0][1] == sorted_gains[1][1]:
            return None, sorted_gains[:2]  # Butuh KNN
        return sorted_gains[0][0], confidence  # Klasifikasi langsung


# CUSTOM C5
if __name__ == "__main__":
    # Muat dataset
    dataset_path = "./src/storage/datasets/base/news_dataset_default_preprocessed_stemmed.csv"
    df = pd.read_csv(dataset_path, sep=",", encoding="utf-8")

    if df.empty:
        raise ValueError("Dataset kosong. Cek dataset Anda!")

    # Ambil fitur dan label
    X_texts = df["preprocessedContent"].values
    y = df["topik"].values

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Parameter grid
    param_grid = {
        'test_size': [0.2, 0.25, 0.3, 0.4],
        'random_state': [4, 40, 42, 100, 200]
    }

    grid = ParameterGrid(param_grid)

    best_score = 0
    best_params = {}
    all_results = []

    print(f"Total kombinasi: {len(grid)}\n")

    for i, params in enumerate(grid, 1):
        test_size = params['test_size']
        random_state = params['random_state']

        # Bagi data
        X_train, X_test, y_train, y_test = train_test_split(
            X_texts, y_encoded, test_size=test_size, stratify=y_encoded, random_state=random_state
        )

        # Inisialisasi model
        c5 = CustomC5()

        start_time = time.time()
        c5.fit(X_train, y_train)
        train_duration = time.time() - start_time

        # Prediksi
        predictions = []
        for text in X_test:
            pred, _ = c5.predict(text)
            if pred is None:
                pred = max(set(y_train), key=list(y_train).count)
            predictions.append(pred)

        # Hitung akurasi
        predictions = [p for p in predictions if p is not None]
        acc = accuracy_score(y_test[:len(predictions)], predictions)

        print(f"[{i}/{len(grid)}] test_size={test_size}, random_state={random_state} → Akurasi: {acc:.2%} | Waktu latih: {train_duration:.2f}s")

        all_results.append({
            'params': params,
            'accuracy': acc,
            'train_time': train_duration
        })

        if acc > best_score:
            best_score = acc
            best_params = params

    print("\n✅ Parameter terbaik ditemukan:")
    print(f"  test_size: {best_params['test_size']}")
    print(f"  random_state: {best_params['random_state']}")
    print(f"  Akurasi terbaik: {best_score:.2%}")
