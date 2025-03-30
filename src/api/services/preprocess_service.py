import os
import pandas as pd
from src.preprocessing.extends.dataset_preprocessor import DatasetPreprocessor
from src.preprocessing.extends.text_preprocessor import TextPreprocessor


class PreprocessService:
    RAW_DATASET_PATH = "src/datasets/raw_dataset.csv"
    PROCESSED_DATASET_PATH = "src/datasets/processed_dataset.csv"

    def __init__(self):
        self.preprocessor = DatasetPreprocessor()
        self.text_preprocessor = TextPreprocessor()

    def preprocess_dataset(self):
        if not os.path.exists(self.RAW_DATASET_PATH):
            return False
        df = self.preprocessor.process(self.RAW_DATASET_PATH)
        df.to_csv(self.PROCESSED_DATASET_PATH, index=False, sep=";")
        return self.PROCESSED_DATASET_PATH

    def fetch_dataset(self, page=1, limit=10, processed=False):
        if not os.path.exists(self.PROCESSED_DATASET_PATH):
            return []

        df = pd.read_csv(self.PROCESSED_DATASET_PATH, sep=";")
        start = (page - 1) * limit
        end = start + limit

        return {
            "data": df.iloc[start:end].to_dict(orient="records"),
            "total_pages": df.shape[0] // limit + 1,
            "current_page": page,
            "limit": limit,
            "total_data": df.shape[0],
            "topic_counts": df["topik"].value_counts().to_dict(),
        }

    def update_label(self, index, new_label):
        df = pd.read_csv(self.PROCESSED_DATASET_PATH, sep=";")
        if index >= len(df):
            return False

        df.at[index, "topik"] = new_label
        df.to_csv(self.PROCESSED_DATASET_PATH, index=False, sep=";")
        return True

    def delete_data(self, index):
        df = pd.read_csv(self.PROCESSED_DATASET_PATH, sep=";")
        if index >= len(df):
            return False

        df = df.drop(index)
        df.to_csv(self.PROCESSED_DATASET_PATH, index=False, sep=";")
        return True

    def add_data(self, contentSnippet, topik):
        """ Menambahkan data baru ke dalam dataset setelah diproses """

        # Preproses teks terlebih dahulu
        preprocessedContent = self.text_preprocessor.preprocess(contentSnippet)

        new_data = pd.DataFrame({
            "contentSnippet": [contentSnippet],
            "preprocessedContent": [preprocessedContent],
            "topik": [topik]
        })

        # Jika file tidak ada, buat dataset baru
        if not os.path.exists(self.PROCESSED_DATASET_PATH):
            new_data.to_csv(self.PROCESSED_DATASET_PATH, index=False, sep=";")
            return True

        # Baca dataset yang sudah ada
        df = pd.read_csv(self.PROCESSED_DATASET_PATH, sep=";")

        # Cek apakah contentSnippet sudah ada
        if df["contentSnippet"].isin(new_data["contentSnippet"]).any():
            return False  # Data sudah ada, tidak perlu ditambahkan

        # Cek apakah preprocessedContent sudah ada
        if df["preprocessedContent"].isin(new_data["preprocessedContent"]).any():
            return False  # Data sudah ada, tidak perlu ditambahkan

        # Tambahkan data baru dan reset index
        df = pd.concat([df, new_data], ignore_index=True)
        df.to_csv(self.PROCESSED_DATASET_PATH, index=False, sep=";")

        return True
