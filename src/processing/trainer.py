import joblib
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from src.models.hybrid import HybridClassifier
from src.preprocessing.preprocessor import Preprocessor


class HybridModelTrainer:
    def __init__(self, dataset_path):
        self.df = Preprocessor.preprocess_dataset(dataset_path)

    def train(self):
        # Membersihkan teks
        self.df["clean_text"] = self.df["contentSnippet"].apply(
            Preprocessor.preprocess_text)

        # Menghapus baris dengan teks kosong setelah preprocessing
        self.df = self.df[self.df["clean_text"].str.strip() != ""]

        X_texts = self.df["clean_text"].tolist()
        y = self.df["topik"]

        X_train, X_test, y_train, y_test = train_test_split(
            X_texts, y, test_size=0.2, stratify=y, random_state=42)

        hybrid_model = HybridClassifier()

        # Latih model
        hybrid_model.fit(X_train, y_train)

        # Prediksi hasil
        y_pred = hybrid_model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
        print("\nConfusion Matrix model Hybrid C5.0-KNN:\n", cm)
        print("\nClassification Report model Hybrid C5.0-KNN:\n",
              classification_report(y_test, y_pred))

        # Menyimpan model ke dalam file
        joblib.dump(hybrid_model, './src/models/saved/hybrid_model.joblib')

        # Menyimpan model ke dalam file
        with open('./src/models/saved/hybrid_model.pkl', 'wb') as file:
            pickle.dump(hybrid_model, file)

        return hybrid_model


if __name__ == "__main__":
    trainer = HybridModelTrainer("./src/dataset/dataset-berita-ppl.csv")
    trainer.train()
