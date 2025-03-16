import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from src.utilities.preprocessor import Preprocessor
from src.utilities.knn import ManualKNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


class HybridClassifier:
    def __init__(self, n_neighbors=7, threshold=0.75):
        self.vectorizer = TfidfVectorizer()
        self.knn = ManualKNN(n_neighbors=n_neighbors)
        self.threshold = threshold
        self.tfidf_matrix = None
        self.labels = None
        self.topics_keywords = {}

    def fit(self, X_train, y_train):
        # Membuat TF-IDF matrix
        self.tfidf_matrix = self.vectorizer.fit_transform(X_train).toarray()
        self.labels = y_train

        # Pelatihan KNN
        self.knn.fit(self.tfidf_matrix, y_train)

        # Menyusun kata-kata terpenting untuk setiap topik
        self.topics_keywords = self.get_top_keywords_for_each_topic(
            X_train, y_train)

    def get_top_keywords_for_each_topic(self, X_train, y_train, top_n=10):
        topic_keywords = {}

        # Untuk setiap kategori (topik), ambil kata-kata dengan TF-IDF tertinggi
        for label in np.unique(y_train):
            indices = [i for i, label_ in enumerate(
                y_train) if label_ == label]
            relevant_docs = [X_train[i] for i in indices]
            tfidf_matrix = self.vectorizer.transform(relevant_docs).toarray()
            feature_names = self.vectorizer.get_feature_names_out()

            # Ambil kata dengan nilai TF-IDF tertinggi
            keyword_scores = np.mean(tfidf_matrix, axis=0)
            top_indices = np.argsort(keyword_scores)[
                ::-1][:top_n]  # Ambil top_n kata
            top_keywords = [feature_names[idx] for idx in top_indices]

            topic_keywords[label] = top_keywords

        return topic_keywords

    def predict(self, X_test):
        X_tfidf = self.vectorizer.transform(X_test).toarray()
        predictions = []

        for vector in X_tfidf:
            # Mencocokkan dokumen dengan setiap node di pohon keputusan (topik)
            predicted_label = self._classify_with_decision_tree(vector)
            if predicted_label is None:
                # Jika tidak ada topik yang cocok, lanjutkan ke KNN
                predicted_label = self.knn.predict([vector])[0]

            predictions.append(predicted_label)

        return predictions

    def _classify_with_decision_tree(self, vector):
        for label, keywords in self.topics_keywords.items():
            # Hitung kecocokan kata-kata dalam vector dengan kata-kata dalam topik
            keyword_matches = sum([vector[self.vectorizer.vocabulary_.get(
                keyword, -1)] > 0 for keyword in keywords])
            match_percentage = keyword_matches / len(keywords)

            if match_percentage >= self.threshold:
                return label  # Jika lebih dari threshold, langsung klasifikasikan ke topik ini

        return None


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
