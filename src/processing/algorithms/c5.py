import pandas as pd
import numpy as np
from collections import Counter
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer


class CustomC5:
    def __init__(self):
        self.topic_data = {}

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


# CUSTOM C5
# if __name__ == "__main__":
#     # Memuat dataset asli
#     dataset_path = "./src/storage/datasets/base/news_dataset_default_preprocessed_stopwords.csv"
#     df = pd.read_csv(dataset_path, sep=",", encoding="utf-8")

#     if df.empty:
#         raise ValueError("Dataset kosong. Cek dataset Anda!")

#     # Ambil fitur (preprocessedContent) dan label (topik) dari dataset
#     X_texts = df["preprocessedContent"].values  # Teks yang sudah diproses
#     y = df["topik"].values  # Label kategori

#     # Tentukan grid parameter untuk pencarian
#     param_grid = {
#         'test_size': [0.2, 0.3, 0.4],  # Pembagian data latih vs uji
#         # Nilai random_state yang berbeda
#         'random_state': [40, 42, 50, 100, 200]
#     }

#     best_score = 0
#     best_params = {}

#     # Lakukan pencarian grid untuk parameter terbaik
#     for test_size in param_grid['test_size']:
#         for random_state in param_grid['random_state']:
#             # Membagi data menjadi data latih dan data uji menggunakan train_test_split
#             X_train, X_test, y_train, y_test = train_test_split(
#                 X_texts, y, test_size=test_size, stratify=y, random_state=random_state
#             )

#             # Inisialisasi dan latih model C5
#             c5 = CustomC5()
#             c5.fit(X_train, y_train)

#             # Prediksi untuk data uji
#             predictions = []
#             for text in X_test:
#                 pred, _ = c5.predict(text)
#                 if pred is None:
#                     # Ganti None dengan label yang lebih sering muncul (label terbanyak)
#                     pred = max(set(y_train), key=list(y_train).count)
#                 predictions.append(pred)

#             # Akurasi (pastikan tidak ada None dalam prediksi)
#             predictions = [p for p in predictions if p is not None]
#             accuracy = accuracy_score(y_test[:len(predictions)], predictions)
#             print(
#                 f"Akurasi model C5 dengan test_size={test_size} dan random_state={random_state}: {accuracy:.2%}")

#             # Simpan parameter terbaik
#             if accuracy > best_score:
#                 best_score = accuracy
#                 best_params = {'test_size': test_size,
#                                'random_state': random_state}

#     print(f"\nParameter terbaik ditemukan: {best_params}")
#     print(f"Akurasi terbaik: {best_score:.2%}")


# LIBRARY C5
if __name__ == "__main__":
    # Memuat dataset asli
    dataset_path = "./src/storage/datasets/base/news_dataset_default_preprocessed_stemmed.csv"
    df = pd.read_csv(dataset_path, sep=",", encoding="utf-8")

    if df.empty:
        raise ValueError("Dataset kosong. Cek dataset Anda!")

    # Ambil fitur (preprocessedContent) dan label (topik) dari dataset
    X = df["preprocessedContent"].values  # Teks yang sudah diproses
    y = df["topik"].values  # Label kategori

    tfidf_vectorizer = TfidfVectorizer(max_features=1000)
    X_tfidf = tfidf_vectorizer.fit_transform(X)

    # Membagi data menjadi data latih dan uji
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=42)

    # Inisialisasi model DecisionTreeClassifier
    dt_model = DecisionTreeClassifier()

    # Menentukan parameter grid untuk GridSearchCV
    param_grid = {
        # Fungsi pembagian (gini atau entropy)
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20, 30],   # Kedalaman maksimum pohon
        # Jumlah minimum sampel untuk membagi simpul
        'min_samples_split': [2, 10, 20],
        # Jumlah minimum sampel di simpul daun
        'min_samples_leaf': [1, 5, 10]
    }

    # Melakukan GridSearchCV
    grid_search = GridSearchCV(estimator=dt_model, param_grid=param_grid,
                               cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    # Tampilkan parameter terbaik dari GridSearch
    print(f"Best parameters: {grid_search.best_params_}")

    # Gunakan model dengan parameter terbaik untuk prediksi
    best_dt_model = grid_search.best_estimator_

    # Prediksi untuk data uji
    predictions = best_dt_model.predict(X_test)

    # Evaluasi model
    accuracy = accuracy_score(y_test, predictions)
    print(f'Akurasi model terbaik: {accuracy:.2%}')
