import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from src.preprocessing.preprocessor import Preprocessor
from src.processing.algorithms.knn import ManualKNN
from src.processing.algorithms.c5 import CustomC5
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


class HybridClassifier:
    def __init__(self, n_neighbors=7, threshold=0.75):
        self.vectorizer = TfidfVectorizer()
        self.knn = ManualKNN(n_neighbors=n_neighbors)
        self.decision_tree = CustomC5(
            self.vectorizer, threshold)
        self.tfidf_matrix = None
        self.labels = None

    def fit(self, X_train, y_train):
        """Melatih model Hybrid dengan KNN dan Decision Tree."""
        self.tfidf_matrix = self.vectorizer.fit_transform(X_train).toarray()
        self.labels = y_train

        self.decision_tree.fit(X_train, y_train)
        self.knn.fit(self.tfidf_matrix, y_train)

    def predict(self, X_test):
        """Melakukan prediksi menggunakan Decision Tree atau KNN."""
        X_tfidf = self.vectorizer.transform(X_test).toarray()
        predictions = []

        for vector in X_tfidf:
            predicted_label = self.decision_tree.classify(vector)
            if predicted_label is None:
                predicted_label = self.knn.predict([vector])[0]

            predictions.append(predicted_label)

        return predictions


if __name__ == "__main__":
    # Load dataset
    df = Preprocessor.preprocess_dataset(
        "./src/dataset/dataset-berita-ppl.csv")

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
