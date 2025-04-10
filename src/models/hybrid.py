import numpy as np
import pandas as pd
from src.utilities.vectorizer import TextVectorizer
from src.processing.algorithms.knn import CustomKNN
from src.processing.algorithms.c5 import CustomC5
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
import pickle


class HybridClassifier:
    def __init__(self, n_neighbors=11, c5_threshold=0.65, max_features=None):
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.c5 = CustomC5()
        self.knn = CustomKNN(
            n_neighbors=n_neighbors, p=2, weights='distance', algorithm="auto")
        self.vectorizer_path = './src/storage/vectorizers/vectorizer.pkl'
        self.is_vectorizer_trained = False
        # Ambang batas untuk memutuskan kapan menggunakan KNN
        self.c5_threshold = c5_threshold

    def fit(self, X_train, y_train, raw_train, le):
        """Melatih C5.0 dan KNN dengan TF-IDF"""
        X_train_vectors = self.vectorizer.fit_transform(X_train)

        # Simpan vectorizer
        with open(self.vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)

        self.is_vectorizer_trained = True  # Tandai vectorizer telah dilatih

        self.c5.fit(X_train, y_train)
        self.knn.fit(X_train_vectors, y_train,
                     original_docs=raw_train, vectorizer=self.vectorizer, label_encoder=le)

    def predict(self, X_test):
        """Memprediksi kategori berdasarkan model Hybrid C5.0-KNN"""
        if not self.is_vectorizer_trained:
            self.vectorizer.load_vectorizer(self.vectorizer_path)

        X_test_vectors = self.vectorizer.transform(X_test)
        predictions = []

        for i, text in enumerate(X_test):
            # C5 prediksi awal dengan confidence
            label, confidence = self.c5.predict(text)

            if label is not None and confidence >= self.c5_threshold:
                # Gunakan C5 jika confidence cukup tinggi
                predictions.append(label)
            else:
                # Gunakan KNN jika confidence rendah atau tidak ada hasil dari C5
                knn_prediction = self.knn.predict(
                    X_test_vectors[i].reshape(1, -1))[0]
                predictions.append(knn_prediction)

        return predictions

    def get_word_stats(self):
        """Mengembalikan dataframe statistik kata dari model C5"""
        return pd.DataFrame(self.c5.word_stats)

    def get_tfidf_word_stats(self, X_docs):
        if not self.is_vectorizer_trained:
            with open(self.vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)

        tfidf_matrix = self.vectorizer.transform(X_docs)
        feature_names = self.vectorizer.get_feature_names_out()
        idf_values = self.vectorizer.idf_

        tfidf_array = tfidf_matrix.toarray()
        total_docs = len(X_docs)
        stats = []

        for idx, word in enumerate(feature_names):
            word_column = tfidf_array[:, idx]
            df = np.count_nonzero(word_column > 0)
            df_ratio = df / total_docs if total_docs > 0 else 0
            avg_tf = np.mean(word_column[word_column > 0]) if df > 0 else 0
            avg_tfidf = np.mean(word_column) if df > 0 else 0
            idf = idf_values[idx]

            stats.append({
                "word": word,
                "df": df,
                "df_ratio": round(df_ratio, 4),
                "tf_avg": round(avg_tf, 4),
                "idf": round(idf, 4),
                "tfidf_avg": round(avg_tfidf, 4),
            })

        return pd.DataFrame(stats).sort_values(by="tfidf_avg", ascending=False).reset_index(drop=True)
