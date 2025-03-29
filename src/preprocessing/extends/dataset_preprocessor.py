import pandas as pd
from src.preprocessing.preprocessor import Preprocessor


class DatasetPreprocessor(Preprocessor):
    def preprocess(self, file_path, sep=";", encoding="utf-8"):
        """ Preprocessing dataset """
        df = pd.read_csv(file_path, sep=sep, encoding=encoding)
        # drop duplikat untuk contentSnippet
        df.drop_duplicates(subset=["contentSnippet"], inplace=True)

        df.dropna(subset=["contentSnippet", "topik"], inplace=True)
        return df
