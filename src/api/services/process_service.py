import os
import json
import uuid
import joblib
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from src.processing.trainer import HybridModelTrainer


class ProcessService:
    STORAGE_PATH = "src/storage/models/trained/"
    METADATA_PATH = "src/storage/metadatas/models.json"

    def __init__(self):
        os.makedirs(self.STORAGE_PATH, exist_ok=True)
        if not os.path.exists(self.METADATA_PATH):
            with open(self.METADATA_PATH, "w") as f:
                json.dump([], f)

    def load_metadata(self):
        with open(self.METADATA_PATH, "r") as f:
            return json.load(f)

    def save_metadata(self, metadata):
        with open(self.METADATA_PATH, "w") as f:
            json.dump(metadata, f, indent=4)

    def split_dataset(self, preprocessed_dataset_path, test_size):
        if not os.path.exists(preprocessed_dataset_path):
            return {}

        df = pd.read_csv(preprocessed_dataset_path, sep=";")
        if df.empty:
            return {}

        X = df["preprocessedContent"]
        y = df["topik"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )

        return {
            "train_size": len(X_train),
            "test_size": len(X_test),
            "train_per_topic": y_train.value_counts().to_dict(),
            "test_per_topic": y_test.value_counts().to_dict()
        }

    def train_model(self, preprocessed_dataset_id, preprocessed_dataset_path, raw_dataset_id, name, n_neighbors, test_size):
        if not os.path.exists(preprocessed_dataset_path):
            return {}

        df = pd.read_csv(preprocessed_dataset_path, sep=";")
        if df.empty:
            return {}

        trainer = HybridModelTrainer(preprocessed_dataset_path)
        hybrid_model, evaluation_results = trainer.train(
            n_neighbors, test_size)
        split_results = self.split_dataset(
            preprocessed_dataset_path, test_size)

        model_id = str(uuid.uuid4())
        model_path = os.path.join(self.STORAGE_PATH, f"{model_id}.joblib")
        joblib.dump(hybrid_model, model_path)  # Simpan model

        metadata = self.load_metadata()

        model_metadata = {
            "id": model_id,
            "name": name,
            "model_path": model_path,
            "preprocessed_dataset_id": preprocessed_dataset_id,
            "raw_dataset_id": raw_dataset_id,
            "total_data": len(df),
            "n_neighbors": n_neighbors,
            "test_size": test_size,
            "train_size": split_results["train_size"],
            "test_size": split_results["test_size"],
            "train_per_topic": split_results["train_per_topic"],
            "test_per_topic": split_results["test_per_topic"],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "evaluation": evaluation_results
        }
        metadata.append(model_metadata)
        self.save_metadata(metadata)

        return model_metadata

    def edit_model_name(self, model_id, new_name):
        metadata = self.load_metadata()
        model_metadata = next(
            (m for m in metadata if m["id"] == model_id), None)
        if not model_metadata:
            return False
        model_metadata["name"] = new_name
        model_metadata["updated_at"] = datetime.now().isoformat()
        self.save_metadata(metadata)
        return True

    def delete_model(self, model_id):
        metadata = self.load_metadata()
        model_metadata = next(
            (m for m in metadata if m["id"] == model_id), None)
        if not model_metadata:
            return False

        # Hapus file model jika ada
        model_path = model_metadata.get("model_path")
        if model_path and os.path.exists(model_path):
            os.remove(model_path)

        metadata = [m for m in metadata if m["id"] != model_id]
        self.save_metadata(metadata)
        return True

    def get_model(self, model_id):
        metadata = self.load_metadata()
        return next((m for m in metadata if m["id"] == model_id), {})

    def get_models(self):
        return self.load_metadata()
