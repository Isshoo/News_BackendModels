# src/api/services/dataset_service.py
import os
import json
import csv
import uuid
import pandas as pd
from datetime import datetime
from src.preprocessing.extends.dataset_preprocessor import DatasetPreprocessor
from src.api.services.preprocess_service import PreprocessService


class DatasetService:
    DATASET_DIR = "src/storage/datasets/uploads"
    METADATA_FILE = "src/storage/metadatas/uploaded_datasets.json"
    HISTORY_FILE = "src/storage/metadatas/dataset_history.json"
    DEFAULT_DATASET_ID = "default-dataset"

    def __init__(self):
        self.preprocessor = DatasetPreprocessor()
        self.preprocess_service = PreprocessService()
        os.makedirs(self.DATASET_DIR, exist_ok=True)
        if not os.path.exists(self.METADATA_FILE):
            with open(self.METADATA_FILE, "w") as f:
                json.dump([], f)
        if not os.path.exists(self.HISTORY_FILE):
            with open(self.HISTORY_FILE, "w") as f:
                json.dump([], f)

        # Ensure default dataset exists
        self._ensure_default_dataset()

    def _ensure_default_dataset(self):
        metadata = self._load_metadata()
        default_exists = any(
            ds['id'] == self.DEFAULT_DATASET_ID for ds in metadata)

        if not default_exists:
            # Create empty default dataset
            dataset_path = os.path.join(
                self.DATASET_DIR, "default_dataset.csv")
            empty_df = pd.DataFrame(columns=["contentSnippet", "topik"])
            empty_df.to_csv(dataset_path, index=False, sep=",",
                            quoting=csv.QUOTE_NONNUMERIC, encoding="utf-8")

            new_entry = {
                "id": self.DEFAULT_DATASET_ID,
                "name": "default_dataset",
                "path": dataset_path,
                "total_data": 0,
                "topic_counts": {},
                "upload_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            metadata.append(new_entry)
            self._save_metadata(metadata)

    def _load_metadata(self):
        with open(self.METADATA_FILE, "r") as f:
            return json.load(f)

    def _save_metadata(self, metadata):
        with open(self.METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=4)

    def _log_history(self, action, dataset_id, details):
        history = self._load_history()
        history.append({
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "dataset_id": dataset_id,
            "details": details
        })
        with open(self.HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=4)

    def _load_history(self):
        with open(self.HISTORY_FILE, "r") as f:
            return json.load(f)

    def save_dataset(self, filepath, dataset_name):
        """Melakukan preprocessing, menyimpan dataset, dan mencatat metadata"""
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
            "updated_at": datetime.now().isoformat()
        }
        metadata.append(new_entry)
        self._save_metadata(metadata)

        # Add non-duplicate data to default dataset
        self._merge_to_default(processed_df, dataset_id)

        return new_entry

    def _merge_to_default(self, new_df, source_dataset_id):
        """Merge non-duplicate data from new dataset to default dataset"""
        default_info = next(
            (d for d in self._load_metadata() if d['id'] == self.DEFAULT_DATASET_ID), None)
        if not default_info:
            return

        default_df = pd.read_csv(default_info["path"], sep=",")

        # Find non-duplicate rows
        mask = ~new_df["contentSnippet"].isin(default_df["contentSnippet"])
        unique_rows = new_df[mask]

        if len(unique_rows) > 0:
            merged_df = pd.concat([unique_rows, default_df], ignore_index=True)
            merged_df.to_csv(default_info["path"], index=False, sep=",",
                             quoting=csv.QUOTE_NONNUMERIC, encoding="utf-8")

            # Update metadata
            metadata = self._load_metadata()
            for ds in metadata:
                if ds['id'] == self.DEFAULT_DATASET_ID:
                    ds['total_data'] = len(merged_df)
                    ds['topic_counts'] = merged_df["topik"].value_counts().to_dict()
                    ds['updated_at'] = datetime.now().isoformat()
                    break
            self._save_metadata(metadata)

            # Log history
            self._log_history(
                "merge_from_upload",
                self.DEFAULT_DATASET_ID,
                {
                    "source_dataset": source_dataset_id,
                    "added_records": len(unique_rows),
                    "topics_added": unique_rows["topik"].value_counts().to_dict()
                }
            )

            # Add non-duplicate data to preprocessed dataset
            data_to_add = [{
                "contentSnippet": row["contentSnippet"],
                "topik": row["topik"]
            } for _, row in unique_rows.iterrows()]

            self.preprocess_service.add_new_data(data_to_add)

    def fetch_datasets(self):
        """Mengambil semua dataset yang tersimpan"""
        metadata = self._load_metadata()
        return sorted(metadata, key=lambda x: x["upload_at"], reverse=True)

    def fetch_dataset(self, dataset_id, page=1, limit=10):
        """Mengambil dataset tertentu dengan paginasi"""
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
        """Menghapus dataset tertentu"""
        metadata = self._load_metadata()
        dataset_info = next(
            (d for d in metadata if d["id"] == dataset_id), None)
        if not dataset_info:
            return False

        os.remove(dataset_info["path"])
        metadata = [d for d in metadata if d["id"] != dataset_id]
        self._save_metadata(metadata)

        return True

    def add_data(self, dataset_id, data_list):
        """Menambahkan data baru ke dataset"""
        metadata = self._load_metadata()
        dataset = next((d for d in metadata if d["id"] == dataset_id), None)

        if not dataset:
            return {"error": "Dataset not found"}, 404

        # if dataset_id == self.DEFAULT_DATASET_ID:
        #     return {"error": "Cannot directly modify default dataset"}, 403

        df = pd.read_csv(dataset["path"], sep=",")
        new_data = pd.DataFrame(data_list)

        # Check for duplicates
        mask = ~new_data["contentSnippet"].isin(df["contentSnippet"])
        unique_data = new_data[mask]

        if len(unique_data) == 0:
            return {"error": "All data already exists"}, 409

        df = pd.concat([unique_data, df], ignore_index=True)
        df.to_csv(dataset["path"], index=False, sep=",",
                  quoting=csv.QUOTE_NONNUMERIC, encoding="utf-8")

        # Update metadata
        for ds in metadata:
            if ds['id'] == dataset_id:
                ds['total_data'] = len(df)
                ds['topic_counts'] = df["topik"].value_counts().to_dict()
                ds['updated_at'] = datetime.now().isoformat()
                break
        self._save_metadata(metadata)

        # Log history
        self._log_history(
            "add_data",
            dataset_id,
            {
                "added_records": len(unique_data),
                "topics_added": unique_data["topik"].value_counts().to_dict()
            }
        )

        # Add non-duplicate data to preprocessed dataset
        data_to_add = [{
            "contentSnippet": row["contentSnippet"],
            "topik": row["topik"]
        } for _, row in unique_data.iterrows()]

        self.preprocess_service.add_new_data(data_to_add)

        return {"message": f"Added {len(unique_data)} new records"}, 201

    def delete_data(self, dataset_id, content_snippets):
        """Menghapus data dari dataset"""
        metadata = self._load_metadata()
        dataset = next((d for d in metadata if d["id"] == dataset_id), None)

        if not dataset:
            return {"error": "Dataset not found"}, 404

        # if dataset_id == self.DEFAULT_DATASET_ID:
        #     return {"error": "Cannot directly modify default dataset"}, 403

        df = pd.read_csv(dataset["path"], sep=",")
        initial_count = len(df)

        # Filter out rows to keep
        df = df[~df["contentSnippet"].isin(content_snippets)]
        removed_count = initial_count - len(df)

        if removed_count == 0:
            return {"error": "No matching records found"}, 404

        df.to_csv(dataset["path"], index=False, sep=",",
                  quoting=csv.QUOTE_NONNUMERIC, encoding="utf-8")

        # Update metadata
        for ds in metadata:
            if ds['id'] == dataset_id:
                ds['total_data'] = len(df)
                ds['topic_counts'] = df["topik"].value_counts().to_dict()
                ds['updated_at'] = datetime.now().isoformat()
                break
        self._save_metadata(metadata)

        # Log history
        self._log_history(
            "delete_data",
            dataset_id,
            {
                "removed_records": removed_count,
                "content_snippets": content_snippets
            }
        )

        return {"message": f"Removed {removed_count} records"}, 200

    def get_history(self, dataset_id=None):
        """Mengambil riwayat perubahan dataset"""
        history = self._load_history()
        if dataset_id:
            history = [h for h in history if h["dataset_id"] == dataset_id]
        return history
