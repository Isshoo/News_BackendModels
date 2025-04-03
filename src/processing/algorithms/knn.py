import pandas as pd
import numpy as np
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder


class CustomKNN:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None
        self.scaler = StandardScaler(with_mean=False)

    def fit(self, X_train, y_train):
        """Menyimpan data latih KNN dan melakukan normalisasi"""
        self.X_train = X_train
        self.y_train = np.array(y_train)
        # Normalisasi fitur
        self.X_train = self.scaler.fit_transform(self.X_train)

    def euclidean_distance(self, vec1, vec2):
        """Menghitung jarak Euclidean antara dua vektor"""
        return np.sqrt(np.sum((vec1 - vec2) ** 2))

    def predict(self, X_test, candidate_labels):
        """Klasifikasi dengan KNN menggunakan kandidat dari C5.0"""
        if self.X_train is None or self.y_train is None:
            raise ValueError(
                "Model belum dilatih. Jalankan `fit()` terlebih dahulu.")

        # Normalisasi data uji menggunakan scaler yang sama yang digunakan pada X_train
        X_test = self.scaler.transform(X_test)  # Transformasi data uji

        y_pred = []
        for test_vector in X_test:
            distances = [
                (i, self.euclidean_distance(test_vector, sample))
                for i, sample in enumerate(self.X_train)
            ]

            distances.sort(key=lambda x: x[1])  # Urutkan berdasarkan jarak
            nearest_labels = [self.y_train[i]
                              for i, _ in distances[:self.n_neighbors]]

            # Pastikan hanya memilih label yang ada di kandidat
            filtered_labels = [
                label for label in nearest_labels if label in candidate_labels]

            y_pred.append(
                Counter(filtered_labels).most_common(1)[0][0] if filtered_labels else Counter(
                    nearest_labels).most_common(1)[0][0]
            )

        return np.array(y_pred)  # Mengembalikan array prediksi


if __name__ == "__main__":
    # Membaca dataset dari file CSV
    dataset_path = "./src/storage/datasets/base/news_dataset_default_preprocessed_stemmed.csv"
    df = pd.read_csv(dataset_path, sep=",", encoding="utf-8")

    if df.empty:
        raise ValueError("Dataset kosong. Cek dataset Anda!")

    # Ambil fitur (preprocessedContent) dan label (topik) dari dataset
    X = df["preprocessedContent"].values  # Teks yang sudah diproses
    y = df["topik"].values  # Label kategori

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Menggunakan TextVectorizer untuk mengubah teks menjadi vektor
    # max_features menentukan jumlah fitur yang diambil
    tfidf_vectorizer = TfidfVectorizer(max_features=1000)
    X_tfidf = tfidf_vectorizer.fit_transform(X)

    # Membagi data menjadi data latih dan data uji menggunakan train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y_encoded, test_size=0.2, stratify=y_encoded, random_state=40
    )

    # Inisialisasi model KNeighborsClassifier
    pipeline = Pipeline([
        ('knn', KNeighborsClassifier())  # Model KNN
    ])

    # Menentukan parameter grid untuk GridSearchCV
    param_grid = {
        'knn__n_neighbors': [3, 5, 7, 9, 11],              # Jumlah tetangga
        # Cara pemberian bobot kepada tetangga
        'knn__weights': ['uniform', 'distance'],
        # Algoritma pencarian tetangga
        'knn__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        # Ukuran daun untuk algoritma tree-based
        'knn__leaf_size': [20, 30, 40],
        # Parameter untuk metric (p=1 adalah Manhattan, p=2 adalah Euclidean)
        'knn__p': [1, 2]
    }

    # Melakukan GridSearchCV
    grid_search = GridSearchCV(pipeline, param_grid=param_grid,
                               cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    # Tampilkan parameter terbaik dari GridSearch
    print(f"Best parameters: {grid_search.best_params_}")

    # Gunakan model dengan parameter terbaik untuk prediksi
    best_knn_model = grid_search.best_estimator_

    # Prediksi untuk data uji
    predictions = best_knn_model.predict(X_test)

    # Evaluasi model
    accuracy = accuracy_score(y_test, predictions)
    print(f'Akurasi model terbaik: {accuracy:.2%}')
