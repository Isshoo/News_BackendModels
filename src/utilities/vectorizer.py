from sklearn.feature_extraction.text import TfidfVectorizer


class TextVectorizer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def fit_transform(self, corpus):
        """Melatih vectorizer dan mengubah teks menjadi vektor"""
        self.vectorizer.fit(corpus)
        # Pastikan .toarray()
        return self.vectorizer.transform(corpus).toarray()

    def transform(self, texts):
        """Mengubah teks baru ke vektor dengan vectorizer yang sudah dilatih"""
        if not hasattr(self.vectorizer, 'vocabulary_'):
            raise ValueError(
                "Vectorizer belum dilatih. Jalankan `fit_transform()` terlebih dahulu.")
        return self.vectorizer.transform(texts).toarray()
