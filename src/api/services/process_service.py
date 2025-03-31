import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from src.processing.trainer import HybridModelTrainer


class ProcessService:
    PROCESSED_DATASET_PATH = "src/storage/datasets/processed/default.csv"
    EVALUATION_PATH = "src/storage/metadatas/models.json"

    def __init__(self):
        self.trainer = HybridModelTrainer(self.PROCESSED_DATASET_PATH)

    def split_dataset(self, test_size):
        """ Membagi dataset ke dalam train dan test set """
        if not os.path.exists(self.PROCESSED_DATASET_PATH):
            return {}

        df = pd.read_csv(self.PROCESSED_DATASET_PATH, sep=";")
        if df.empty:
            return {}

        X = df["preprocessedContent"]
        y = df["topik"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42)

        train_counts = y_train.value_counts().to_dict()
        test_counts = y_test.value_counts().to_dict()

        return {
            "train_size": len(X_train),
            "test_size": len(X_test),
            "train_per_topic": train_counts,
            "test_per_topic": test_counts
        }

    def train_model(self, n_neighbors, test_size):
        """ Melatih model dengan parameter yang dipilih """
        if not os.path.exists(self.PROCESSED_DATASET_PATH):
            return {}

        df = pd.read_csv(self.PROCESSED_DATASET_PATH, sep=";")
        if df.empty:
            return {}

        results = self.trainer.train(n_neighbors, test_size)

        # Simpan hasil evaluasi ke file JSON
        evaluation_data = {
            "message": "Model trained successfully",
            "evaluation": results
        }

        with open(self.EVALUATION_PATH, "w") as f:
            json.dump(evaluation_data, f, indent=4)

        return evaluation_data

    def model_evaluation(self):
        """ Mengambil hasil evaluasi model yang telah disimpan """
        if not os.path.exists(self.EVALUATION_PATH):
            return {}

        with open(self.EVALUATION_PATH, "r") as f:
            evaluation_data = json.load(f)

        return evaluation_data["evaluation"]
