import pandas as pd
import csv
from datetime import datetime
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

        df.drop_duplicates(subset=["preprocessedContent"], inplace=True)
        df.dropna(subset=["preprocessedContent"], inplace=True)

        df["is_preprocessed"] = True
        df["is_trained"] = False
        df["inserted_at"] = datetime.now().isoformat()
        df["updated_at"] = datetime.now().isoformat()

        return df

    def raw_process(self, file_path, sep=",", encoding="utf-8"):
        """ Preprocessing dataset """
        df = pd.read_csv(file_path, sep=sep, encoding=encoding)

        # Tambahkan kolom preprocessing text
        df["preprocessedContent"] = df["contentSnippet"].apply(
            self.text_preprocessor.preprocess)

        df.drop_duplicates(subset=["preprocessedContent"], inplace=True)
        df.dropna(subset=["preprocessedContent"], inplace=True)

        df.to_csv(file_path, index=False, sep=",",
                  quoting=csv.QUOTE_NONNUMERIC, encoding="utf-8")

        return df

    def raw_formatter(self, file_path="./src/storage/datasets/base/raw_news_dataset.xlsx"):
        # Baca file Excel
        df = pd.read_excel(file_path)

        # Ganti tanda petik dua dalam kolom contentSnippet menjadi petik satu
        df['contentSnippet'] = df['contentSnippet'].str.replace('"', "'")

        df.drop_duplicates(subset=["contentSnippet"], inplace=True)
        df.dropna(subset=["contentSnippet"], inplace=True)

        # Simpan sebagai CSV dengan format yang benar
        df.to_csv(file_path,
                  index=False, quoting=csv.QUOTE_NONNUMERIC, encoding="utf-8")

    def raw_combiner(self, file_path="./src/storage/datasets/base/DataTesting2.xlsx"):
        # Baca file Excel
        df = pd.read_excel(file_path)

        # Ganti tanda petik dua dalam kolom contentSnippet menjadi petik satu
        df['contentSnippet'] = df['judul'] + ' - ' + df['contentSnippet']
        # drop judul
        df.drop("judul", axis=1, inplace=True)
        df['contentSnippet'] = df['contentSnippet'].str.replace('"', "'")

        df.drop_duplicates(subset=["contentSnippet"], inplace=True)
        df.dropna(subset=["contentSnippet"], inplace=True)

        # Simpan sebagai CSV dengan format yang benar
        df.to_csv("./src/storage/datasets/base/DataTesting2.csv",
                  index=False, quoting=csv.QUOTE_NONNUMERIC, encoding="utf-8")


if __name__ == "__main__":
    preprocessor = DatasetPreprocessor()
    preprocessor.raw_process("./src/storage/datasets/base/DataTesting.csv")
