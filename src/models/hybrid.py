import numpy as np
from src.utilities.vectorizer import TextVectorizer
from src.processing.algorithms.knn import CustomKNN
from src.processing.algorithms.c5 import CustomC5
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from src.preprocessing.extends.dataset_preprocessor import DatasetPreprocessor
from src.preprocessing.extends.text_preprocessor import TextPreprocessor


class HybridClassifier:
    def __init__(self, n_neighbors=5):
        self.vectorizer = TextVectorizer()
        self.c5 = CustomC5()
        self.knn = CustomKNN(n_neighbors)

    def fit(self, X_train, y_train):
        """Melatih C5.0 dan KNN dengan TF-IDF"""
        X_train_vectors = self.vectorizer.fit_transform(X_train)
        self.c5.fit(X_train, y_train)
        self.knn.fit(X_train_vectors, y_train)

    def predict(self, X_test):
        """Memprediksi kategori berdasarkan model Hybrid C5.0-KNN"""
        predictions = []
        X_test_vectors = self.vectorizer.transform(X_test)

        for i, text in enumerate(X_test):
            label, candidates = self.c5.predict(text)  # Pakai C5 dulu
            if label is not None:
                # Jika cukup informasi, gunakan hasil C5
                predictions.append(label)
            else:
                candidate_labels = [c[0]
                                    for c in candidates]  # Label kandidat dari C5
                predictions.append(self.knn.predict(
                    X_test_vectors[i], candidate_labels))  # Pakai KNN

        return predictions


if __name__ == "__main__":
    dataset_preprocessor = DatasetPreprocessor()
    text_preprocessor = TextPreprocessor()

    # Memuat dataset
    df = dataset_preprocessor.preprocess(
        "./src/datasets/dataset-berita-ppl.csv")

    df["clean_text"] = text_preprocessor.preprocess(
        df["contentSnippet"].tolist())

    # Menghapus baris dengan teks kosong setelah preprocessing
    df = df[df["clean_text"].str.strip() != ""]

    X_texts = df["clean_text"].tolist()
    y = df["topik"]

    X_train, X_test, y_train, y_test = train_test_split(
        X_texts, y, test_size=0.2, stratify=y, random_state=42)

    hybrid_model = HybridClassifier()

    # Latih model
    hybrid_model.fit(X_train, y_train)

    # Prediksi hasil
    y_pred = hybrid_model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
    print("\nConfusion Matrix model Hybrid C5.0-KNN:\n", cm)
    print("\nClassification Report model Hybrid C5.0-KNN:\n",
          classification_report(y_test, y_pred))

    # Input sample text
    sample_text = input("Masukkan berita yang akan diklasifikasi: ")
    prediction = hybrid_model.predict([sample_text])[0]

    if prediction == "ekonomi":
        prediction = "Ekonomi"
    elif prediction == "teknologi":
        prediction = "Teknologi"
    elif prediction == "olahraga":
        prediction = "Olahraga"
    elif prediction == "hiburan":
        prediction = "Hiburan"
    elif prediction == "gayahidup":
        prediction = "Gaya Hidup"

    print(f"Hasil klasifikasi: {prediction}")
