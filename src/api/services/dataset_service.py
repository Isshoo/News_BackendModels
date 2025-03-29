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

        return df.iloc[start:end].to_dict(orient="records")

    def count_dataset(self):
        """ Menghitung jumlah total data dalam dataset """
        if not os.path.exists(self.DATASET_PATH):
            return 0

        df = pd.read_csv(self.DATASET_PATH, sep=";")
        return len(df)

    def get_topics_distribution(self):
        """ Mendapatkan daftar topik dan jumlahnya dalam dataset """
        if not os.path.exists(self.DATASET_PATH):
            return {}

        df = pd.read_csv(self.DATASET_PATH, sep=";")
        topic_counts = df["topik"].value_counts(
        ).to_dict()  # Hitung jumlah setiap topik

        return topic_counts
