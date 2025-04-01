from src.features.classifier import NewsClassifier


class PredictService:

    def predict(self, text, model_path):
        """ Mengklasifikasikan teks berita menggunakan model hybrid dan DeepSeek """
        classifier = NewsClassifier(model_path)

        result = classifier.classify(text)

        return result
