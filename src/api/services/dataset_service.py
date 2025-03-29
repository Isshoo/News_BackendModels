import os
import pandas as pd
from src.preprocessing.extends.dataset_preprocessor import DatasetPreprocessor

DATASET_PATH = "src/datasets/processed_dataset.csv"


def save_dataset(filepath):
    preprocessor = DatasetPreprocessor()
    df = preprocessor.preprocess(filepath, sep=";")

    # Simpan dataset hasil preprocessing
    df.to_csv(DATASET_PATH, index=False, sep=";")

    return DATASET_PATH


def fetch_dataset(page, limit):
    if not os.path.exists(DATASET_PATH):
        return []

    df = pd.read_csv(DATASET_PATH, sep=";")

    start = (page - 1) * limit
    end = start + limit

    return df.iloc[start:end].to_dict(orient="records")


def count_dataset():
    if not os.path.exists(DATASET_PATH):
        return 0

    df = pd.read_csv(DATASET_PATH, sep=";")

    return len(df)
