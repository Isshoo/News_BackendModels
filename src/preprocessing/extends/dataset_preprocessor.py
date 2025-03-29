import pandas as pd
from src.preprocessing.preprocessor import Preprocessor
from src.preprocessing.extends.text_preprocessor import TextPreprocessor


class DatasetPreprocessor(Preprocessor):
    def __init__(self):
        self.text_preprocessor = TextPreprocessor()

    def preprocess(self, file_path, sep=";", encoding="utf-8"):
        """ Preprocessing dataset """
        df = pd.read_csv(file_path, sep=sep, encoding=encoding)
        # drop duplikat untuk contentSnippet
        df.drop_duplicates(subset=["contentSnippet"], inplace=True)

        df.dropna(subset=["contentSnippet", "topik"], inplace=True)
        return df

    def process(self, file_path, sep=";", encoding="utf-8"):
        """ Preprocessing dataset """
        df = pd.read_csv(file_path, sep=sep, encoding=encoding)

        # Tambahkan kolom preprocessing text
        df["preprocessedContent"] = df["contentSnippet"].apply(
            self.text_preprocessor.preprocess)

        return df
