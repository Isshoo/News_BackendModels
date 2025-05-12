# src/api/services/preprocess_service.py
import os
import json
import string
import csv
import uuid
import pandas as pd
from datetime import datetime
from src.preprocessing.extends.dataset_preprocessor import DatasetPreprocessor
from src.preprocessing.extends.text_preprocessor import TextPreprocessor


class PreprocessService:
    PREPROCESSED_DIR = "src/storage/datasets/preprocessed"
    METADATA_FILE = "src/storage/metadatas/preprocessed_datasets.json"
    DEFAULT_DATASET_ID = "default-preprocessed"

    def __init__(self):
        os.makedirs(self.PREPROCESSED_DIR, exist_ok=True)
        if not os.path.exists(self.METADATA_FILE):
            with open(self.METADATA_FILE, "w") as f:
                json.dump([], f)
        self.text_preprocessor = TextPreprocessor()
        self.dataset_preprocessor = DatasetPreprocessor()
        self._ensure_default_preprocessed_dataset()

    def _ensure_default_preprocessed_dataset(self):
        """Membuat default preprocessed dataset jika belum ada"""
        metadata = self.load_metadata()
        default_exists = any(
            ds['id'] == self.DEFAULT_DATASET_ID for ds in metadata)

        if not default_exists:
            # Create empty default dataset
            dataset_path = os.path.join(
                self.PREPROCESSED_DIR, "default_preprocessed.csv")
            empty_df = pd.DataFrame(columns=[
                "contentSnippet",
                "topik",
                "preprocessedContent",
                "is_preprocessed",
                "is_trained",
                "inserted_at",
                "updated_at"
            ])
            empty_df.to_csv(dataset_path, index=False, sep=",",
                            quoting=csv.QUOTE_NONNUMERIC, encoding="utf-8")

            new_entry = {
                "id": self.DEFAULT_DATASET_ID,
                "raw_dataset_id": "default-dataset",
                "path": dataset_path,
                "name": "default",
                "total_data": 0,
                "total_preprocessed": 0,
                "total_trained": 0,
                "topic_counts": {},
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            metadata.append(new_entry)
            self.save_metadata(metadata)

    def load_metadata(self):
        with open(self.METADATA_FILE, "r") as f:
            return json.load(f)

    def save_metadata(self, metadata):
        with open(self.METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=4)

    def _update_metadata_stats(self, dataset_id):
        """Update statistics in metadata"""
        metadata = self.load_metadata()
        dataset_info = next(
            (d for d in metadata if d["id"] == dataset_id), None)
        if not dataset_info:
            return

        df = pd.read_csv(dataset_info["path"], sep=",")
        dataset_info["total_data"] = len(df)
        dataset_info["total_preprocessed"] = len(
            df[df["is_preprocessed"] == True])
        dataset_info["total_trained"] = len(df[df["is_trained"] == True])
        dataset_info["topic_counts"] = df["topik"].value_counts().to_dict()
        dataset_info["updated_at"] = datetime.now().isoformat()

        self.save_metadata(metadata)

    def add_new_data(self, data_list):
        """Menambahkan data baru ke default preprocessed dataset"""
        metadata = self.load_metadata()
        dataset = next(
            (d for d in metadata if d["id"] == self.DEFAULT_DATASET_ID), None)
        if not dataset:
            return {"error": "Default preprocessed dataset not found"}, 404

        df = pd.read_csv(dataset["path"], sep=",")

        # Prepare new data
        new_data = pd.DataFrame(data_list)
        new_data["preprocessedContent"] = ""
        new_data["is_preprocessed"] = False
        new_data["is_trained"] = False
        new_data["inserted_at"] = datetime.now().isoformat()
        new_data["updated_at"] = datetime.now().isoformat()

        # Check for duplicates
        mask = ~new_data["contentSnippet"].isin(df["contentSnippet"])
        unique_data = new_data[mask]

        if len(unique_data) == 0:
            return {"error": "All data already exists"}, 409

        # Concatenate and save
        df = pd.concat([unique_data, df], ignore_index=True)
        df.to_csv(dataset["path"], index=False, sep=",",
                  quoting=csv.QUOTE_NONNUMERIC, encoding="utf-8")

        # Update metadata
        self._update_metadata_stats(self.DEFAULT_DATASET_ID)

        return {"message": f"Added {len(unique_data)} new records"}, 201

    def preprocess_new_data(self):
        """Melakukan preprocessing pada data baru yang belum diproses"""
        metadata = self.load_metadata()
        dataset = next(
            (d for d in metadata if d["id"] == self.DEFAULT_DATASET_ID), None)
        if not dataset:
            return {"error": "Default preprocessed dataset not found"}, 404

        df = pd.read_csv(dataset["path"], sep=",")

        # Get unprocessed data
        unprocessed = df[df["is_preprocessed"] == False]
        if len(unprocessed) == 0:
            return {"message": "No new data to preprocess"}, 200

        # Process each unprocessed row
        for idx, row in unprocessed.iterrows():
            preprocessed_content = self.text_preprocessor.preprocess(
                row["contentSnippet"])
            if preprocessed_content:
                df.at[idx, "preprocessedContent"] = preprocessed_content
                df.at[idx, "is_preprocessed"] = True
                df.at[idx, "updated_at"] = datetime.now().isoformat()

        # Save changes
        df.to_csv(dataset["path"], index=False, sep=",",
                  quoting=csv.QUOTE_NONNUMERIC, encoding="utf-8")

        # Update metadata
        self._update_metadata_stats(self.DEFAULT_DATASET_ID)

        return {"message": f"Preprocessed {len(unprocessed)} new records"}, 200

    def fetch_preprocessed_data(self, page=1, limit=10, filter_type="all"):
        """Mengambil data dari default preprocessed dataset dengan filter"""
        metadata = self.load_metadata()
        dataset = next(
            (d for d in metadata if d["id"] == self.DEFAULT_DATASET_ID), None)
        if not dataset:
            return {"error": "Default preprocessed dataset not found"}, 404

        df = pd.read_csv(dataset["path"], sep=",")

        # Apply filters
        if filter_type == "old":
            df = df[df["is_trained"] == True]
        elif filter_type == "new":
            df = df[df["is_trained"] == False]
        elif filter_type == "unprocessed":
            df = df[df["is_preprocessed"] == False]
        elif filter_type == "processed":
            df = df[df["is_preprocessed"] == True]

        # Pagination
        start = (page - 1) * limit
        end = start + limit
        total_data = len(df)

        return {
            "data": df.iloc[start:end].reset_index().to_dict(orient="records"),
            "total_data": total_data,
            "total_pages": (total_data + limit - 1) // limit,
            "current_page": page,
            "limit": limit,
            "stats": {
                "total_old": len(df[df["is_trained"] == True]),
                "total_new": len(df[df["is_trained"] == False]),
                "total_preprocessed": len(df[df["is_preprocessed"] == True]),
                "total_unprocessed": len(df[df["is_preprocessed"] == False]),
                "topic_counts": df["topik"].value_counts().to_dict()
            }
        }

    def edit_new_data(self, index, new_label=None, new_content=None):
        """Mengedit data baru (yang belum di-train)"""
        metadata = self.load_metadata()
        dataset = next(
            (d for d in metadata if d["id"] == self.DEFAULT_DATASET_ID), None)
        if not dataset:
            return {"error": "Default preprocessed dataset not found"}, 404

        df = pd.read_csv(dataset["path"], sep=",")
        if index >= len(df):
            return {"error": "Data not found"}, 404

        # Hanya bisa edit data yang belum di-train
        if df.at[index, "is_trained"]:
            return {"error": "Cannot edit trained data"}, 403

        if new_label:
            df.at[index, "topik"] = new_label
        if new_content:
            df.at[index, "preprocessedContent"] = new_content

        df.at[index, "updated_at"] = datetime.now().isoformat()
        df.to_csv(dataset["path"], index=False, sep=",",
                  quoting=csv.QUOTE_NONNUMERIC, encoding="utf-8")

        self._update_metadata_stats(self.DEFAULT_DATASET_ID)
        return {"message": "Data updated successfully"}, 200

    def delete_new_data(self, indices):
        """Menghapus data baru (yang belum di-train)"""
        metadata = self.load_metadata()
        dataset = next(
            (d for d in metadata if d["id"] == self.DEFAULT_DATASET_ID), None)
        if not dataset:
            return {"error": "Default preprocessed dataset not found"}, 404

        df = pd.read_csv(dataset["path"], sep=",")

        # Filter hanya data yang belum di-train
        mask = (df.index.isin(indices)) & (df["is_trained"] == False)
        to_delete = df[mask]

        if len(to_delete) == 0:
            return {"error": "No deletable data found (only new/un-trained data can be deleted)"}, 404

        df = df[~mask]
        df.to_csv(dataset["path"], index=False, sep=",",
                  quoting=csv.QUOTE_NONNUMERIC, encoding="utf-8")

        self._update_metadata_stats(self.DEFAULT_DATASET_ID)
        return {"message": f"Deleted {len(to_delete)} records"}, 200

    def mark_data_as_trained(self, indices):
        """Menandai data sebagai sudah di-train"""
        metadata = self.load_metadata()
        dataset = next(
            (d for d in metadata if d["id"] == self.DEFAULT_DATASET_ID), None)
        if not dataset:
            return {"error": "Default preprocessed dataset not found"}, 404

        df = pd.read_csv(dataset["path"], sep=",")

        # Pastikan data sudah diproses
        unprocessed = df[df.index.isin(indices) & (
            df["is_preprocessed"] == False)]
        if len(unprocessed) > 0:
            return {"error": f"{len(unprocessed)} data are not preprocessed yet"}, 400

        df.loc[df.index.isin(indices), "is_trained"] = True
        df.loc[df.index.isin(indices),
               "updated_at"] = datetime.now().isoformat()
        df.to_csv(dataset["path"], index=False, sep=",",
                  quoting=csv.QUOTE_NONNUMERIC, encoding="utf-8")

        self._update_metadata_stats(self.DEFAULT_DATASET_ID)
        return {"message": f"Marked {len(indices)} records as trained"}, 200
