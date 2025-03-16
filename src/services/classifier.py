from src.utilities.preprocessor import Preprocessor
from src.models.deepseek import DeepSeekClassifier
import joblib


def classify_sample(model, sample_text):
    processed_sample_text = Preprocessor.preprocess_text(sample_text)
    print(f"\nPrediksi topik dari teks: \"{sample_text}\"\n")

    hasil_model_hybrid = model.predict([processed_sample_text])[0]

    if hasil_model_hybrid == "ekonomi":
        hasil_model_hybrid = "Ekonomi"
    elif hasil_model_hybrid == "teknologi":
        hasil_model_hybrid = "Teknologi"
    elif hasil_model_hybrid == "olahraga":
        hasil_model_hybrid = "Olahraga"
    elif hasil_model_hybrid == "hiburan":
        hasil_model_hybrid = "Hiburan"
    elif hasil_model_hybrid == "gayahidup":
        hasil_model_hybrid = "Gaya Hidup"

    print("Hybrid C5.0-KNN  :", hasil_model_hybrid)
    print("DeepSeek         :", DeepSeekClassifier.classify(
        processed_sample_text, use_api=True))


if __name__ == "__main__":
    import sys
    sys.path.append('./src')
    # Load model hybrid
    hybrid_model = joblib.load('./src/models/saved/hybrid_model.joblib')

    # Input sample text
    sample_text = input("Masukkan berita yang akan diklasifikasi: ")

    # Klasifikasi sample text
    classify_sample(hybrid_model, sample_text)
