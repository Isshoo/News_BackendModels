import os
import pandas as pd
from src.preprocessing.extends.dataset_preprocessor import DatasetPreprocessor


class DatasetService:
    DATASET_PATH = "src/datasets/raw_dataset.csv"

    def __init__(self):
        self.preprocessor = DatasetPreprocessor()

    def save_dataset(self, filepath):
        """ Melakukan preprocessing dan menyimpan dataset yang telah diproses """
        df = self.preprocessor.preprocess(filepath, sep=";")
        df.to_csv(self.DATASET_PATH, index=False, sep=";")
        return self.DATASET_PATH

    def fetch_dataset(self, page=1, limit=10):
        """ Mengambil dataset dengan paginasi """
        if not os.path.exists(self.DATASET_PATH):
            return []

        df = pd.read_csv(self.DATASET_PATH, sep=";")

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
