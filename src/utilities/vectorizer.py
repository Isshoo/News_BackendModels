from sklearn.feature_extraction.text import TfidfVectorizer


class TextVectorizer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def fit_transform(self, texts):
        self.vectors = self.vectorizer.fit_transform(texts).toarray()
        return self.vectors

    def transform(self, texts):
        return self.vectorizer.transform(texts).toarray()
