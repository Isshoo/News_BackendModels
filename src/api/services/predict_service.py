from src.features.classifier import NewsClassifier


class PredictService:
    def __init__(self):
        # TODO: Implement TextPreprocessor class for preprocessing text data
        self.classifier = NewsClassifier()

    def predict(self, text):
        """ Mengklasifikasikan teks berita menggunakan model hybrid dan DeepSeek """
        result = self.classifier.classify(text)

        return result
