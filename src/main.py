from src.utilities.preprocessor import Preprocessor
from src.services.training import train_hybrid_model
from src.services.classifier import classify_sample
import joblib
import sys


def main():
    sys.path.append('./src')

    pilihan = input(
        "Apakah anda ingin melakukan klasifikasi dengan melatih model baru? (y/n) : ")
    if pilihan == "y":
        # Load dataset dan preprocess
        df = Preprocessor.preprocess_dataset(
            "./src/dataset/dataset-berita-ppl.csv")

        # Latih model dan dapatkan hasilnya
        hybrid_model = train_hybrid_model(df)
    else:
        # Load model hybrid
        hybrid_model = joblib.load('./src/models/saved/hybrid_model.joblib')

    # Klasifikasi untuk input sample
    sample_text = input("\nMasukkan berita yang akan diklasifikasi: ")
    classify_sample(hybrid_model, sample_text)


if __name__ == "__main__":
    main()
