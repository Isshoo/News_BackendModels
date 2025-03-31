import os
import json
import uuid
import pandas as pd
from datetime import datetime
from src.preprocessing.extends.dataset_preprocessor import DatasetPreprocessor
from src.preprocessing.extends.text_preprocessor import TextPreprocessor


class PreprocessService:
    DATASET_DIR = "src/storage/datasets/uploads"
    METADATA_FILE = "src/storage/metadatas/preprocessed_datasets.json"

    def __init__(self):
        os.makedirs(self.DATASET_DIR, exist_ok=True)
        if not os.path.exists(self.METADATA_FILE):
            with open(self.METADATA_FILE, "w") as f:
                json.dump([], f)
        self.text_preprocessor = TextPreprocessor()
        self.dataset_preprocessor = DatasetPreprocessor()

    def load_metadata(self):
        with open(self.METADATA_FILE, "r") as f:
            return json.load(f)

    def save_metadata(self, metadata):
        with open(self.METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=4)

    def preprocess_dataset(self, raw_dataset_id, raw_dataset_path, raw_dataset_name):
        if not os.path.exists(raw_dataset_path):
            return False

        df = self.dataset_preprocessor.process(raw_dataset_path)
        processed_path = os.path.join(
            self.DATASET_DIR, f"{raw_dataset_name}_original_preprocessed.csv")
        df.to_csv(processed_path, index=False, sep=";")

        metadata = self.load_metadata()
        new_entry = {
            "id": raw_dataset_id,
            "raw_dataset_id": raw_dataset_id,
            "path": processed_path,
            "name": "default",
            "total_data": len(df),
            "topic_counts": df["topik"].value_counts().to_dict(),
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        metadata.append(new_entry)
        self.save_metadata(metadata)

        return new_entry

    def create_preprocessed_copy(self, raw_dataset_id, name):
        # load metadata for get default dataset
        metadata = self.load_metadata()
        default_entry = next(
            (d for d in metadata if d["id"] == raw_dataset_id), None)
        if not default_entry:
            return False

        raw_dataset_id = default_entry["raw_dataset_id"]
        raw_dataset_path = default_entry["path"]
        raw_dataset_name = os.path.basename(raw_dataset_path).split("_")[0]

        # create a copy of the default dataset

        dataset_id = str(uuid.uuid4())
        copy_path = os.path.join(
            self.DATASET_DIR, f"{raw_dataset_name}_{name}.csv")
        df = pd.read_csv(default_entry["path"], sep=";")
        df.to_csv(copy_path, index=False, sep=";")

        metadata = self.load_metadata()
        new_entry = {
            "id": dataset_id,
            "raw_dataset_id": raw_dataset_id,
            "path": copy_path,
            "name": name,
            "total_data": len(df),
            "topic_counts": df["topik"].value_counts().to_dict(),
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        metadata.append(new_entry)
        self.save_metadata(metadata)

        return new_entry

    def fetch_preprocessed_datasets(self, raw_dataset_id):
        return [d for d in self.load_metadata() if d["raw_dataset_id"] == raw_dataset_id]

    def fetch_preprocessed_dataset(self, dataset_id, page=1, limit=10):
        metadata = self.load_metadata()
        dataset_info = next(
            (d for d in metadata if d["id"] == dataset_id), None)
        if not dataset_info:
            return None

        df = pd.read_csv(dataset_info["path"], sep=";")
        start = (page - 1) * limit
        end = start + limit

        return {
            "data": df.iloc[start:end].reset_index().to_dict(orient="records"),
            "total_data": len(df),
            "total_pages": (len(df) + limit - 1) // limit,
            "current_page": page,
            "limit": limit,
            "topic_counts": dataset_info["topic_counts"],
        }

    def delete_preprocessed_dataset(self, dataset_id):
        metadata = self.load_metadata()
        entry = next(
            (d for d in metadata if d["id"] == dataset_id and d["name"] != "default"), None)
        if not entry:
            return False

        os.remove(entry["path"])
        metadata = [d for d in metadata if d["id"] != dataset_id]
        self.save_metadata(metadata)
        return True

    def update_label(self, dataset_id, index, new_label):
        metadata = self.load_metadata()
        entry = next(
            (d for d in metadata if d["id"] == dataset_id and d["name"] != "default"), None)
        if not entry:
            return False

        df = pd.read_csv(entry["path"], sep=";")
        if index >= len(df):
            return False

        df.at[index, "topik"] = new_label
        return self.update_preprocessed_dataset(dataset_id, df)

    def delete_data(self, dataset_id, index):
        metadata = self.load_metadata()
        entry = next(
            (d for d in metadata if d["id"] == dataset_id and d["name"] != "default"), None)
        if not entry:
            return False

        df = pd.read_csv(entry["path"], sep=";")
        if index >= len(df):
            return False

        df = df.drop(index).reset_index(drop=True)
        return self.update_preprocessed_dataset(dataset_id, df)

    def add_data(self, dataset_id, contentSnippet, topik):
        metadata = self.load_metadata()
        entry = next(
            (d for d in metadata if d["id"] == dataset_id and d["name"] != "default"), None)
        if not entry:
            return False

        df = pd.read_csv(entry["path"], sep=";")
        preprocessedContent = self.text_preprocessor.preprocess(contentSnippet)
        new_data = pd.DataFrame({
            "contentSnippet": [contentSnippet],
            "preprocessedContent": [preprocessedContent],
            "topik": [topik]
        })

        if df["contentSnippet"].isin(new_data["contentSnippet"]).any() or df["preprocessedContent"].isin(new_data["preprocessedContent"]).any():
            return False

        df = pd.concat([df, new_data], ignore_index=True)
        return self.update_preprocessed_dataset(dataset_id, df)

    def update_preprocessed_dataset(self, dataset_id, df):
        metadata = self.load_metadata()
        entry = next(
            (d for d in metadata if d["id"] == dataset_id and d["name"] != "default"), None)
        if not entry:
            return False

        df.to_csv(entry["path"], index=False, sep=";")
        entry["total_data"] = len(df)
        entry["topic_counts"] = df["topik"].value_counts().to_dict()
        entry["updated_at"] = datetime.now().isoformat()

        self.save_metadata(metadata)
        return True
