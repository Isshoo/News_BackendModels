import pandas as pd
from sklearn.model_selection import train_test_split
from src.models.hybrid import HybridClassifier
from src.utilities.model_evaluations import evaluate_model
from src.utilities.save_model import save_model


class HybridModelTrainer:
    def __init__(self, df,):
        """Inisialisasi trainer dengan dataset dalam bentuk DataFrame"""

        self.df = pd.read_csv(df, sep=";", encoding="utf-8")

        if self.df.empty:
            raise ValueError("Dataset kosong. Cek dataset Anda!")

    def train(self, n_neighbors, test_size):
        """Melatih model Hybrid C5.0-KNN"""

        X_texts = self.df["preprocessedContent"]
        y = self.df["topik"]

        X_train, X_test, y_train, y_test = train_test_split(
            X_texts, y, test_size=test_size, stratify=y, random_state=42
        )

        hybrid_model = HybridClassifier(n_neighbors)

        # Latih model
        hybrid_model.fit(X_train, y_train)

        # Prediksi hasil
        y_pred = hybrid_model.predict(X_test)

        # Evaluasi model
        evaluate_results = evaluate_model(y_test, y_pred)

        # Simpan model
        save_model(hybrid_model)

        return evaluate_results


if __name__ == "__main__":
    df = "./src/datasets/dataset-berita-ppl.csv"
    trainer = HybridModelTrainer(df)
    trainer.train()
