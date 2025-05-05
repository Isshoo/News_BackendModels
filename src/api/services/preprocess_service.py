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
    PROCESSED_DIR = "src/storage/datasets/processed"
    METADATA_FILE = "src/storage/metadatas/preprocessed_datasets.json"

    def __init__(self):
        os.makedirs(self.PREPROCESSED_DIR, exist_ok=True)
        os.makedirs(self.PROCESSED_DIR, exist_ok=True)
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
            self.PREPROCESSED_DIR, f"{raw_dataset_name}_original_preprocessed.csv")
        df.to_csv(processed_path, index=False,
                  quoting=csv.QUOTE_NONNUMERIC, encoding="utf-8")

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
            self.PROCESSED_DIR, f"{raw_dataset_name}_{name}_processed.csv")
        df = pd.read_csv(default_entry["path"], sep=",")
        df.to_csv(copy_path, index=False, sep=",",
                  quoting=csv.QUOTE_NONNUMERIC, encoding="utf-8")

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

    def fetch_all_preprocessed_datasets(self):
        return self.load_metadata()

    def fetch_preprocessed_datasets(self, raw_dataset_id):
        return [d for d in self.load_metadata() if d["raw_dataset_id"] == raw_dataset_id]

    def fetch_preprocessed_dataset(self, dataset_id, page=1, limit=10):
        metadata = self.load_metadata()
        dataset_info = next(
            (d for d in metadata if d["id"] == dataset_id), None)
        if not dataset_info:
            return False

        df = pd.read_csv(dataset_info["path"], sep=",")
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

    def delete_preprocessed_dataset(self, dataset_id, raw_dataset_id=''):
        metadata = self.load_metadata()
        dataset = next((d for d in metadata if d["id"] == dataset_id), None)

        if not dataset:
            return {"error": "Preprocessed dataset not found"}, 404

        if dataset_id == 'default-stemming':
            return False

        # jika fungsi dipanggil dengan parameter raw_dataset_id maka langsung terhapus
        if raw_dataset_id == dataset["raw_dataset_id"]:
            os.remove(dataset["path"])
            metadata = [d for d in metadata if d["id"] != dataset_id]
            self.save_metadata(metadata)
            return {"error": "Preprocessed dataset deleted successfully by Raw Dataset"}, 200

        if dataset["name"] == "default":
            return {"error": "Default preprocessed dataset cannot be deleted"}, 403

        os.remove(dataset["path"])
        metadata = [d for d in metadata if d["id"] != dataset_id]
        self.save_metadata(metadata)

        return {"message": "Preprocessed dataset deleted successfully"}, 200

    def update_data(self, dataset_id, index, new_label, new_preprocessed_content):
        metadata = self.load_metadata()
        dataset = next((d for d in metadata if d["id"] == dataset_id), None)

        if not dataset:
            return {"error": "Preprocessed dataset not found"}, 404

        if dataset["name"] == "default":
            return {"error": "Default preprocessed dataset cannot be edited"}, 403

        df = pd.read_csv(dataset["path"], sep=",")
        if index >= len(df):
            return {"error": "Data not found"}, 404

        # cek jika data hanya 1 huruf
        if len(new_preprocessed_content) == 1:
            return {"error": "Data must be at least 2 characters"}, 400

        for word in new_preprocessed_content.split():
            if word.isdigit():
                return {"error": f"Data cannot contain word with only number: '{word}'"}, 400
            elif word in string.punctuation:
                return {"error": f"Data cannot contain word with only punctuation character: '{word}'"}, 400
            elif all(char in string.punctuation for char in word):
                return {"error": f"Data cannot contain word with only punctuation characters: '{word}'"}, 400

        df.at[index, "topik"] = new_label
        df.at[index, "preprocessedContent"] = new_preprocessed_content

        result = self.update_preprocessed_dataset(dataset_id, df)

        if not result:
            return {"error": "Failed to update data"}, 500
        return {"message": "Data updated successfully"}, 200

    def delete_data(self, dataset_id, index):
        metadata = self.load_metadata()
        dataset = next((d for d in metadata if d["id"] == dataset_id), None)

        if not dataset:
            return {"error": "Preprocessed dataset not found"}, 404

        if dataset["name"] == "default":
            return {"error": "Default preprocessed dataset cannot be edited"}, 403

        df = pd.read_csv(dataset["path"], sep=",")
        if index >= len(df):
            return {"error": "Data not found"}, 404

        df = df.drop(index).reset_index(drop=True)
        result = self.update_preprocessed_dataset(dataset_id, df)
        if not result:
            return {"error": "Failed to delete data"}, 500
        return {"message": "Data deleted successfully"}, 200

    def add_data(self, dataset_id, contentSnippet, topik):
        metadata = self.load_metadata()
        dataset = next((d for d in metadata if d["id"] == dataset_id), None)

        if not dataset:
            return {"error": "Preprocessed dataset not found"}, 404

        if dataset["name"] == "default":
            return {"error": "Default preprocessed dataset cannot be edited"}, 403

        df = pd.read_csv(dataset["path"], sep=",")
        preprocessedContent = self.text_preprocessor.preprocess(contentSnippet)
        if preprocessedContent is None:
            return {"error": "Content is empty after preprocessing"}, 500
        new_data = pd.DataFrame({
            "contentSnippet": [contentSnippet],
            "preprocessedContent": [preprocessedContent],
            "topik": [topik]
        })

        if df["contentSnippet"].isin(new_data["contentSnippet"]).any() or df["preprocessedContent"].isin(new_data["preprocessedContent"]).any():
            return {"error": "Data already exists"}, 409

        df = pd.concat([new_data, df], ignore_index=True)
        result = self.update_preprocessed_dataset(dataset_id, df)
        if not result:
            return {"error": "Failed to add data"}, 500
        return {"message": "Data added successfully"}, 201

    def update_preprocessed_dataset(self, dataset_id, df):
        metadata = self.load_metadata()
        dataset = next((d for d in metadata if d["id"] == dataset_id), None)

        if not dataset:
            return False

        if dataset["name"] == "default":
            return False

        df.to_csv(dataset["path"], index=False, sep=",",
                  quoting=csv.QUOTE_NONNUMERIC, encoding="utf-8")
        dataset["total_data"] = len(df)
        dataset["topic_counts"] = df["topik"].value_counts().to_dict()
        dataset["updated_at"] = datetime.now().isoformat()

        self.save_metadata(metadata)
        return True
