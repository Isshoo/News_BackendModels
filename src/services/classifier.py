from src.preprocessing.preprocessor import Preprocessor
from src.models.deepseek import DeepSeekClassifier
from src.utilities.map_hybrid_result import map_hybrid_result
import joblib


class NewsClassifier:
    def __init__(self, hybrid_model):
        self.hybrid_model = joblib.load(hybrid_model)

    def classify(self, sample_text):
        processed_sample_text = Preprocessor.preprocess_text(sample_text)
        print(f"\nPrediksi topik dari teks: \"{sample_text}\"\n")

        hasil_model_hybrid = self.hybrid_model.predict(
            [processed_sample_text])[0]
        hasil_model_hybrid = map_hybrid_result(hasil_model_hybrid)

        hasil_deepseek = DeepSeekClassifier.classify(
            processed_sample_text, use_api=True)

        print("Hybrid C5.0-KNN  :", hasil_model_hybrid)
        print("DeepSeek         :", hasil_deepseek)

        return {
            "Hybrid C5.0-KNN": hasil_model_hybrid,
            "DeepSeek": hasil_deepseek
        }


if __name__ == "__main__":
    import sys
    sys.path.append('./src')

    classifier = NewsClassifier('./src/models/saved/hybrid_model.joblib')
    sample_text = input("Masukkan berita yang akan diklasifikasi: ")
    classifier.classify(sample_text)
