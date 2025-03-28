import numpy as np


class CustomC5:
    def __init__(self, vectorizer, threshold=0.75):
        self.vectorizer = vectorizer
        self.threshold = threshold
        self.topics_keywords = {}

    def fit(self, X_train, y_train):
        """Menyusun kata-kata terpenting untuk setiap topik."""
        self.topics_keywords = self.get_top_keywords_for_each_topic(
            X_train, y_train)

    def get_top_keywords_for_each_topic(self, X_train, y_train, top_n=10):
        """Mengambil kata kunci teratas untuk setiap topik berdasarkan TF-IDF."""
        topic_keywords = {}

        for label in np.unique(y_train):
            indices = [i for i, label_ in enumerate(
                y_train) if label_ == label]
            relevant_docs = [X_train[i] for i in indices]
            tfidf_matrix = self.vectorizer.transform(relevant_docs).toarray()
            feature_names = self.vectorizer.get_feature_names_out()

            keyword_scores = np.mean(tfidf_matrix, axis=0)
            top_indices = np.argsort(keyword_scores)[::-1][:top_n]
            top_keywords = [feature_names[idx] for idx in top_indices]

            topic_keywords[label] = top_keywords

        return topic_keywords

    def classify(self, vector):
        """Mencocokkan dokumen dengan setiap topik berdasarkan kata kunci."""
        for label, keywords in self.topics_keywords.items():
            keyword_matches = sum([vector[self.vectorizer.vocabulary_.get(
                keyword, -1)] > 0 for keyword in keywords])
            match_percentage = keyword_matches / len(keywords)

            if match_percentage >= self.threshold:
                return label  # Klasifikasikan ke topik ini jika cocok

        return None
