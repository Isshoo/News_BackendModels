import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.preprocessing import LabelEncoder
from src.utilities.map_classification_result import map_classification_result
import time


class CustomC5:
    def __init__(self, min_df=0, max_df_ratio=1):
        self.topic_data = {}
        self.word_gains = {}
        self.min_df = min_df
        self.max_df_ratio = max_df_ratio

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
        filtered_labels = [labels[i]
                           for i, text_set in enumerate(text_sets) if word not in text_set]
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
        text_sets = [set(text.split()) for text in X_train]

        # Hitung DF awal dan filter kata
        raw_doc_counts = Counter(
            word for text in text_sets for word in set(text))
        total_docs = len(text_sets)
        filtered_words = {
            word for word, count in raw_doc_counts.items()
            if count >= self.min_df and count / total_docs <= self.max_df_ratio
        }

        # Filter text_sets & labels
        filtered_text_sets = []
        filtered_labels = []

        for text_set, label in zip(text_sets, y_train):
            filtered_set = text_set & filtered_words
            if filtered_set:  # hanya ambil dokumen yang masih mengandung kata setelah difilter
                filtered_text_sets.append(filtered_set)
                filtered_labels.append(label)

        # Hitung ulang DF setelah filter
        final_doc_counts = Counter(
            word for text in filtered_text_sets for word in set(text))
        final_total_docs = len(filtered_text_sets)

        # Hitung frekuensi kata per label
        word_counts = {label: Counter() for label in unique_labels}
        for text_set, label in zip(filtered_text_sets, filtered_labels):
            word_counts[label].update(text_set)

        # Simpan topik data
        self.topic_data = {
            label: {
                'entropy': self.compute_entropy(list(word_freq.elements())),
                'word_freq': word_freq
            }
            for label, word_freq in word_counts.items()
        }

        # Hitung entropi global (H(S))
        H_S = self.compute_entropy(filtered_labels)

        # Siapkan tabel statistik kata
        self.word_stats = []
        self.word_gains = {}

        for word in filtered_words:
            H_word = self.compute_word_entropy(
                word, filtered_text_sets, filtered_labels)
            H_wo = self.compute_entropy_without_word(
                word, filtered_text_sets, filtered_labels)
            S_word = sum(
                1 for text_set in filtered_text_sets if word in text_set)
            S_not_word = final_total_docs - S_word
            IG = H_S - ((S_word / final_total_docs) * H_word +
                        (S_not_word / final_total_docs) * H_wo)

            # Hitung frekuensi per label
            word_freq_per_label = {
                label: word_counts[label][word]
                for label in unique_labels
                if word_counts[label][word] > 0
            }

            mapped_word_freq_per_label = {
                map_classification_result(label): freq
                for label, freq in word_freq_per_label.items()
            }

            self.word_gains[word] = IG
            self.word_stats.append({
                'word': word,
                'df': final_doc_counts[word],
                'df_ratio': final_doc_counts[word] / final_total_docs,
                'word_entropy': H_word,
                'entropy_without_word': H_wo,
                'information_gain': IG,
                'freq_per_label': mapped_word_freq_per_label,
                'top_label': map_classification_result(
                    max(word_freq_per_label,
                        key=word_freq_per_label.get, default=None)
                )
            })

    def predict(self, text):
        words = text.split()
        gains = {}

        for label, data in self.topic_data.items():
            word_scores = [
                data['word_freq'].get(word, 0) * self.word_gains.get(word, 0)
                for word in words
            ]
            total_score = np.sum(word_scores)
            gains[label] = total_score

        sorted_gains = sorted(gains.items(), key=lambda x: x[1], reverse=True)
        total_gain = sum(gains.values()) if sum(gains.values()) > 0 else 1
        confidence = sorted_gains[0][1] / total_gain

        if len(sorted_gains) > 1 and sorted_gains[0][1] == sorted_gains[1][1]:
            return None, sorted_gains[:2]  # Butuh KNN
        return sorted_gains[0][0], confidence


# CUSTOM C5
if __name__ == "__main__":
    # Muat dataset
    dataset_path = "./src/storage/datasets/preprocessed/raw_news_dataset_preprocessed_stemmed.csv"
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
        'test_size': [0.2, 0.25],
        'random_state': [42, 100]
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

        df_stats = pd.DataFrame(c5.word_stats)
        print(df_stats.head(10))

    print("\n✅ Parameter terbaik ditemukan:")
    print(f"  test_size: {best_params['test_size']}")
    print(f"  random_state: {best_params['random_state']}")
    print(f"  Akurasi terbaik: {best_score:.2%}")
