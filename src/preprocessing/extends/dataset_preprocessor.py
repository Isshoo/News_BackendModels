import pandas as pd
import csv
from src.preprocessing.preprocessor import Preprocessor
from src.preprocessing.extends.text_preprocessor import TextPreprocessor


class DatasetPreprocessor(Preprocessor):
    def __init__(self):
        self.text_preprocessor = TextPreprocessor()

    def preprocess(self, file_path, sep=",", encoding="utf-8"):
        """ Preprocessing dataset """
        df = pd.read_csv(file_path, sep=sep, encoding=encoding)

        # kolom harus sesuai, cek jika terdapat kolom yang diperlukan
        required_columns = {"contentSnippet", "topik"}
        if not required_columns.issubset(df.columns):
            raise ValueError(
                f"File CSV harus memiliki kolom: {', '.join(required_columns)}")

        # drop duplikat untuk contentSnippet
        df.drop_duplicates(subset=["contentSnippet"], inplace=True)

        df.dropna(subset=["contentSnippet", "topik"], inplace=True)

        df['contentSnippet'] = df['contentSnippet'].str.replace('"', "'")

        return df

    def process(self, file_path, sep=",", encoding="utf-8"):
        """ Preprocessing dataset """
        df = pd.read_csv(file_path, sep=sep, encoding=encoding)

        # Tambahkan kolom preprocessing text
        df["preprocessedContent"] = df["contentSnippet"].apply(
            self.text_preprocessor.preprocess)

        return df

    def raw_formatter(self, file_path="./src/storage/datasets/base/news_dataset_default.xlsx"):
        # Baca file Excel
        df = pd.read_excel(file_path)

        # Ganti tanda petik dua dalam kolom contentSnippet menjadi petik satu
        df['contentSnippet'] = df['contentSnippet'].str.replace('"', "'")

        # Simpan sebagai CSV dengan format yang benar
        df.to_csv("./src/storage/datasets/base/news_dataset_default.csv",
                  index=False, quoting=csv.QUOTE_NONNUMERIC, encoding="utf-8")


if __name__ == "__main__":
    preprocessor = DatasetPreprocessor()
    preprocessor.raw_formatter()
