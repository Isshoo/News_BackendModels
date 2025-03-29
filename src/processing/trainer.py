import time
import joblib
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from src.models.hybrid import HybridClassifier
from src.preprocessing.extends.dataset_preprocessor import DatasetPreprocessor
from src.preprocessing.extends.text_preprocessor import TextPreprocessor


class HybridModelTrainer:
    def __init__(self, dataset_path):
        self.dataset_preprocessor = DatasetPreprocessor()
        self.text_preprocessor = TextPreprocessor()

        start_time = time.time()  # Waktu mulai preprocessing
        self.df = self.dataset_preprocessor.preprocess(dataset_path)
        end_time = time.time()  # Waktu selesai preprocessing

        print(
            f"Preprocessing dataset selesai dalam {end_time - start_time:.2f} detik")

        if self.df.empty:
            raise ValueError(
                "Dataset kosong setelah preprocessing. Cek dataset Anda!")

    def train(self):
        # Membersihkan teks
        start_time = time.time()  # Waktu mulai preprocessing teks
        self.df["clean_text"] = self.text_preprocessor.preprocess(
            self.df["contentSnippet"].tolist()
        )
        end_time = time.time()  # Waktu selesai preprocessing teks
        print(
            f"Preprocessing teks selesai dalam {end_time - start_time:.2f} detik")

        # Menghapus baris dengan teks kosong setelah preprocessing
        self.df = self.df[self.df["clean_text"].str.strip() != ""]

        if self.df.empty:
            raise ValueError(
                "Semua data kosong setelah preprocessing. Periksa preprocessing Anda.")

        print("Jumlah data sebelum split:", len(self.df))

        X_texts = self.df["clean_text"].tolist()
        y = self.df["topik"]

        X_train, X_test, y_train, y_test = train_test_split(
            X_texts, y, test_size=0.2, stratify=y, random_state=42
        )

        hybrid_model = HybridClassifier()

        # Latih model
        start_time = time.time()  # Waktu mulai training
        hybrid_model.fit(X_train, y_train)
        end_time = time.time()  # Waktu selesai training
        print(
            f"Training model selesai dalam {end_time - start_time:.2f} detik")

        # Prediksi hasil
        start_time = time.time()  # Waktu mulai prediksi
        y_pred = hybrid_model.predict(X_test)
        end_time = time.time()  # Waktu selesai prediksi
        print(f"Prediksi selesai dalam {end_time - start_time:.2f} detik")

        cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
        print("\nConfusion Matrix model Hybrid C5.0-KNN:\n", cm)
        print("\nClassification Report model Hybrid C5.0-KNN:\n",
              classification_report(y_test, y_pred))

        # Menyimpan model ke dalam file
        joblib.dump(hybrid_model, './src/models/saved/hybrid_model.joblib')

        with open('./src/models/saved/hybrid_model.pkl', 'wb') as file:
            pickle.dump(hybrid_model, file)

        return hybrid_model


if __name__ == "__main__":
    trainer = HybridModelTrainer("./src/datasets/dataset-berita-ppl.csv")
    trainer.train()
