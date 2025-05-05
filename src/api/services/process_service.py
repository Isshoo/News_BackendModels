import os
import json
import shutil
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

        df = pd.read_csv(preprocessed_dataset_path, sep=",")

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

    def train_model(self, preprocessed_dataset_id, preprocessed_dataset_path, raw_dataset_id, name, n_neighbors, split_size):

        df = pd.read_csv(preprocessed_dataset_path, sep=",")

        trainer = HybridModelTrainer(preprocessed_dataset_path)
        hybrid_model, evaluation_results, word_stats_df, tfidf_stats, df_neighbors, df_predict_results = trainer.train(
            n_neighbors, split_size)
        split_results = self.split_dataset(
            preprocessed_dataset_path, split_size)

        model_id = str(uuid.uuid4())
        model_path = os.path.join(self.STORAGE_PATH, f"{model_id}.joblib")
        joblib.dump(hybrid_model, model_path)  # Simpan model

        # Simpan metadata umum ke models.json
        metadata = self.load_metadata()
        model_metadata = {
            "id": model_id,
            "name": name,
            "model_path": model_path,
            "preprocessed_dataset_id": preprocessed_dataset_id,
            "raw_dataset_id": raw_dataset_id,
            "total_data": len(df),
            "n_neighbors": n_neighbors,
            "split_size": split_size,
            "accuracy": evaluation_results["accuracy"],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        metadata.append(model_metadata)
        self.save_metadata(metadata)

        # Simpan metadata tambahan per model
        model_meta_dir = os.path.join("src/storage/metadatas/models", model_id)
        os.makedirs(model_meta_dir, exist_ok=True)

        # Parameter
        parameters = {
            "n_neighbors": n_neighbors,
            "split_size": split_size,
            "train_size": split_results["train_size"],
            "test_size": split_results["test_size"],
            "train_per_topic": split_results["train_per_topic"],
            "test_per_topic": split_results["test_per_topic"]
        }
        with open(os.path.join(model_meta_dir, "parameters.json"), "w") as f:
            json.dump(parameters, f, indent=4)

        # Evaluation
        with open(os.path.join(model_meta_dir, "evaluation.json"), "w") as f:
            json.dump(evaluation_results, f, indent=4)

        # Word Stats
        word_stats_df.to_csv(os.path.join(
            model_meta_dir, "word_stats.csv"), index=False)

        # TF-IDF Stats
        tfidf_stats.to_csv(os.path.join(
            model_meta_dir, "tfidf_stats.csv"), index=False)

        # Neighbors
        df_neighbors.to_csv(os.path.join(
            model_meta_dir, "neighbors.csv"), index=False)

        # Predict Results
        df_predict_results.to_csv(os.path.join(
            model_meta_dir, "predict_results.csv"), index=False)

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

        if model_id == 'default-stemmed':
            return False  # Default model cannot be deleted

        # Hapus file model jika ada
        model_path = model_metadata.get("model_path")
        if model_path and os.path.exists(model_path):
            os.remove(model_path)

        # Hapus folder metadata model
        model_dir = os.path.join("src/storage/metadatas/models", model_id)
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)

        metadata = [m for m in metadata if m["id"] != model_id]
        self.save_metadata(metadata)
        return True

    def get_models(self):
        metadata = self.load_metadata()
        return sorted(metadata, key=lambda x: x["created_at"], reverse=True)

    def get_model(self, model_id):
        metadata = self.load_metadata()
        return next((m for m in metadata if m["id"] == model_id), {})

    def get_word_stats(self, model_id, page=1, limit=10):
        model_dir = os.path.join("src/storage/metadatas/models", model_id)
        file_path = os.path.join(model_dir, "word_stats.csv")
        if not os.path.exists(file_path):
            return None

        df = pd.read_csv(file_path)
        start = (page - 1) * limit
        end = start + limit

        return {
            "data": df.iloc[start:end].to_dict(orient="records"),
            "total_data": len(df),
            "total_pages": (len(df) + limit - 1) // limit,
            "current_page": page,
            "limit": limit,
            "initial_entropy": df["initial_entropy"].mean() if "initial_entropy" in df.columns else None
        }

    def tfidf_stats(self, model_id, page=1, limit=10):
        model_dir = os.path.join("src/storage/metadatas/models", model_id)
        file_path = os.path.join(model_dir, "tfidf_stats.csv")
        if not os.path.exists(file_path):
            return None

        df = pd.read_csv(file_path)
        start = (page - 1) * limit
        end = start + limit

        return {
            "data": df.iloc[start:end].to_dict(orient="records"),
            "total_data": len(df),
            "total_pages": (len(df) + limit - 1) // limit,
            "current_page": page,
            "limit": limit,
        }

    def neighbors(self, model_id, page=1, limit=10):
        model_dir = os.path.join("src/storage/metadatas/models", model_id)
        file_path = os.path.join(model_dir, "neighbors.csv")
        if not os.path.exists(file_path):
            return None

        df = pd.read_csv(file_path)
        start = (page - 1) * limit
        end = start + limit

        return {
            "data": df.iloc[start:end].to_dict(orient="records"),
            "total_data": len(df),
            "total_pages": (len(df) + limit - 1) // limit,
            "current_page": page,
            "limit": limit,
        }

    def predict_results(self, model_id, page=1, limit=10, predict_by=None):
        model_dir = os.path.join("src/storage/metadatas/models", model_id)
        file_path = os.path.join(model_dir, "predict_results.csv")
        if not os.path.exists(file_path):
            return None

        df = pd.read_csv(file_path)

        # Hitung total C5.0 dan KNN
        total_c5 = len(df[df["predict_by"] == "C5.0 Decision"])
        total_knn = len(df[df["predict_by"].str.startswith("KNN")])

        # Filter by prediction method
        if predict_by == "knn":
            df = df[df["predict_by"].str.contains("KNN", na=False)]
        elif predict_by == "c5":
            df = df[df["predict_by"] == "C5.0 Decision"]

        start = (page - 1) * limit
        end = start + limit

        return {
            "data": df.iloc[start:end].to_dict(orient="records"),
            "total_data": len(df),
            "total_c5": total_c5,
            "total_knn": total_knn,
            "total_pages": (len(df) + limit - 1) // limit,
            "current_page": page,
            "limit": limit,
            "predict_by": predict_by
        }

    def get_parameters(self, model_id):
        model_dir = os.path.join("src/storage/metadatas/models", model_id)
        param_path = os.path.join(model_dir, "parameters.json")
        if not os.path.exists(param_path):
            return None
        with open(param_path, "r") as f:
            return json.load(f)

    def get_evaluation(self, model_id):
        model_dir = os.path.join("src/storage/metadatas/models", model_id)
        eval_path = os.path.join(model_dir, "evaluation.json")
        if not os.path.exists(eval_path):
            return None
        with open(eval_path, "r") as f:
            return json.load(f)
