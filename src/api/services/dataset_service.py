import os
import json
import csv
import uuid
import pandas as pd
from datetime import datetime
from src.preprocessing.extends.dataset_preprocessor import DatasetPreprocessor


class DatasetService:
    DATASET_DIR = "src/storage/datasets/uploads"
    METADATA_FILE = "src/storage/metadatas/uploaded_datasets.json"

    def __init__(self):
        self.preprocessor = DatasetPreprocessor()
        os.makedirs(self.DATASET_DIR, exist_ok=True)
        if not os.path.exists(self.METADATA_FILE):
            with open(self.METADATA_FILE, "w") as f:
                json.dump([], f)

    def _load_metadata(self):
        with open(self.METADATA_FILE, "r") as f:
            return json.load(f)

    def _save_metadata(self, metadata):
        with open(self.METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=4)

    def save_dataset(self, filepath, dataset_name):
        """ Melakukan preprocessing, menyimpan dataset, dan mencatat metadata """
        dataset_id = str(uuid.uuid4())
        processed_df = self.preprocessor.preprocess(filepath, sep=",")
        dataset_path = os.path.join(self.DATASET_DIR, f"{dataset_name}.csv")
        processed_df.to_csv(dataset_path, index=False, sep=",",
                            quoting=csv.QUOTE_NONNUMERIC, encoding="utf-8")

        metadata = self._load_metadata()
        new_entry = {
            "id": dataset_id,
            "name": dataset_name,
            "path": dataset_path,
            "total_data": len(processed_df),
            "topic_counts": processed_df["topik"].value_counts().to_dict(),
            "upload_at": datetime.now().isoformat(),
        }
        metadata.append(new_entry)
        self._save_metadata(metadata)

        return new_entry

    def fetch_datasets(self):
        """ Mengambil semua dataset yang tersimpan """
        metadata = self._load_metadata()
        return sorted(metadata, key=lambda x: x["upload_at"], reverse=True)

    def fetch_dataset(self, dataset_id, page=1, limit=10):
        """ Mengambil dataset tertentu dengan paginasi """
        metadata = self._load_metadata()
        dataset_info = next(
            (d for d in metadata if d["id"] == dataset_id), None)
        if not dataset_info:
            return None

        df = pd.read_csv(dataset_info["path"], sep=",")
        start = (page - 1) * limit
        end = start + limit

        return {
            "data": df.iloc[start:end].reset_index().to_dict(orient="records"),
            "total_pages": (len(df) + limit - 1) // limit,
            "current_page": page,
            "limit": limit,
            "total_data": len(df),
            "topic_counts": dataset_info["topic_counts"],
        }

    def delete_dataset(self, dataset_id):
        """ Menghapus dataset tertentu """
        metadata = self._load_metadata()
        dataset_info = next(
            (d for d in metadata if d["id"] == dataset_id), None)
        if not dataset_info:
            return False

        os.remove(dataset_info["path"])
        metadata = [d for d in metadata if d["id"] != dataset_id]
        self._save_metadata(metadata)

        return True

    def add_data(self, dataset_id, contentSnippet, topik):
        metadata = self.load_metadata()
        dataset = next((d for d in metadata if d["id"] == dataset_id), None)

        if not dataset:
            return {"error": "Preprocessed dataset not found"}, 404

        if dataset["name"] == "default":
            return {"error": "Default preprocessed dataset cannot be edited"}, 403

        df = pd.read_csv(dataset["path"], sep=",")
        new_data = pd.DataFrame({
            "contentSnippet": [contentSnippet],
            "topik": [topik]
        })

        if df["contentSnippet"].isin(new_data["contentSnippet"]).any():
            return {"error": "Data already exists"}, 409

        df = pd.concat([new_data, df], ignore_index=True)
        result = self.update_preprocessed_dataset(dataset_id, df)
        if not result:
            return {"error": "Failed to add data"}, 500
        return {"message": "Data added successfully"}, 201
