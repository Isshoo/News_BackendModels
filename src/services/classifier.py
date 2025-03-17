from src.utilities.preprocessor import Preprocessor
from src.models.deepseek import DeepSeekClassifier


class NewsClassifier:
    def __init__(self, hybrid_model):
        self.hybrid_model = hybrid_model

    def classify(self, sample_text):
        processed_sample_text = Preprocessor.preprocess_text(sample_text)
        print(f"\nPrediksi topik dari teks: \"{sample_text}\"\n")

        hasil_model_hybrid = self.hybrid_model.predict(
            [processed_sample_text])[0]
        hasil_model_hybrid = self._map_hybrid_result(hasil_model_hybrid)

        hasil_deepseek = DeepSeekClassifier.classify(
            processed_sample_text, use_api=True)

        print("Hybrid C5.0-KNN  :", hasil_model_hybrid)
        print("DeepSeek         :", hasil_deepseek)

        return {
            "Hybrid C5.0-KNN": hasil_model_hybrid,
            "DeepSeek": hasil_deepseek
        }

    @staticmethod
    def _map_hybrid_result(result):
        mapping = {
            "ekonomi": "Ekonomi",
            "teknologi": "Teknologi",
            "olahraga": "Olahraga",
            "hiburan": "Hiburan",
            "gayahidup": "Gaya Hidup"
        }
        return mapping.get(result, result)


if __name__ == "__main__":
    import sys
    sys.path.append('./src')

    classifier = NewsClassifier('./src/models/saved/hybrid_model.joblib')
    sample_text = input("Masukkan berita yang akan diklasifikasi: ")
    classifier.classify(sample_text)
