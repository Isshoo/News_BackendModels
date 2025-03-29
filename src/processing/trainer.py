import time
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from src.models.hybrid import HybridClassifier
from src.preprocessing.extends.text_preprocessor import TextPreprocessor


class HybridModelTrainer:
    def __init__(self, df, n_neighbors=5, train_test_size=0.2):
        """Inisialisasi trainer dengan dataset dalam bentuk DataFrame"""
        self.text_preprocessor = TextPreprocessor()
        self.n_neighbors = n_neighbors
        self.train_test_size = train_test_size

        self.df = pd.read_csv(df, sep=";", encoding="utf-8")

        if self.df.empty:
            raise ValueError("Dataset kosong. Cek dataset Anda!")

    def preprocess_data(self):
        """Membersihkan teks dan mempersiapkan data untuk pelatihan"""
        start_time = time.time()
        self.df["clean_text"] = self.df["contentSnippet"].apply(
            self.text_preprocessor.preprocess)
        end_time = time.time()
        print(f"Preprocessing selesai dalam {end_time - start_time:.2f} detik")

        # Menghapus baris yang kosong setelah preprocessing
        self.df = self.df[self.df["clean_text"].str.strip() != ""]
        if self.df.empty:
            raise ValueError("Semua data kosong setelah preprocessing.")

        return self.df

    def train(self):
        """Melatih model Hybrid C5.0-KNN"""
        self.preprocess_data()
        print("Jumlah data sebelum split:", len(self.df))

        X_texts = self.df["clean_text"].tolist()
        y = self.df["topik"]

        X_train, X_test, y_train, y_test = train_test_split(
            X_texts, y, test_size=self.train_test_size, stratify=y, random_state=42
        )
        print("Jumlah data setelah split:", len(X_train), len(X_test))

        hybrid_model = HybridClassifier(self.n_neighbors)

        # Latih model
        start_time = time.time()
        hybrid_model.fit(X_train, y_train)
        end_time = time.time()
        print(f"Training selesai dalam {end_time - start_time:.2f} detik")

        # Prediksi hasil
        start_time = time.time()
        y_pred = hybrid_model.predict(X_test)
        end_time = time.time()
        print(f"Prediksi selesai dalam {end_time - start_time:.2f} detik")

        # Evaluasi model
        self.evaluate_model(y_test, y_pred)

        # Simpan model
        self.save_model(hybrid_model)

        return hybrid_model

    def evaluate_model(self, y_test, y_pred):
        """Evaluasi model dengan confusion matrix dan classification report"""
        labels = np.unique(y_test)
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        print("\nConfusion Matrix:\n", cm)
        print("\nClassification Report:\n",
              classification_report(y_test, y_pred))

    def save_model(self, model):
        """Menyimpan model dalam format joblib"""
        model_path = './src/models/saved/hybrid_model.joblib'
        joblib.dump(model, model_path)
        print(f"Model berhasil disimpan di {model_path}")


if __name__ == "__main__":
    df = "./src/datasets/dataset-berita-ppl.csv"
    trainer = HybridModelTrainer(df)
    trainer.train()
