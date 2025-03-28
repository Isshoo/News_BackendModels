from src.services.classifier import NewsClassifier
from src.processing.trainer import HybridModelTrainer
import joblib
import sys

sys.path.append('./src')


def main():

    pilihan = input(
        "Apakah anda ingin melakukan klasifikasi dengan melatih model baru? (y/n) : ")
    if pilihan == "y":
        # Load dataset dan preprocess
        df = "./src/dataset/dataset-berita-ppl.csv"

        # Latih model dan dapatkan hasilnya
        trainer = HybridModelTrainer(df)
        hybrid_model = trainer.train()
    else:
        # Load model hybrid
        hybrid_model = joblib.load('./src/models/saved/hybrid_model.joblib')

    # Klasifikasi untuk input sample
    sample_text = input("\nMasukkan berita yang akan diklasifikasi: ")
    classifier = NewsClassifier(hybrid_model)
    classifier.classify(sample_text)


if __name__ == "__main__":
    main()
