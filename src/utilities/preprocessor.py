import re
import string
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Pastikan NLTK sudah mengunduh resource yang dibutuhkan
'''
import ssl
import nltk
ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
'''


# Inisialisasi stemmer bahasa Indonesia
factory = StemmerFactory()
stemmer = factory.create_stemmer()


class Preprocessor:

    @staticmethod
    def preprocess_dataset(file_path, sep=";", encoding="utf-8"):
        """ Preprocessing keseluruhan dataset """

        df = pd.read_csv(file_path, sep=sep, encoding=encoding)

        # Menghapus duplikasi
        df.drop_duplicates(inplace=True)

        # Menangani missing values
        df.dropna(subset=["contentSnippet", "topik"], inplace=True)

        # Membersihkan teks
        df["clean_text"] = df["contentSnippet"].apply(
            Preprocessor.preprocess_text)

        # Menghapus baris dengan teks kosong setelah preprocessing
        df = df[df["clean_text"].str.strip() != ""]

        return df

    @staticmethod
    def preprocess_text(text):
        """ Preprocessing teks """

        # Case Folding (konversi ke huruf kecil)
        text = text.lower()

        # Menghapus tanda baca, angka, dan karakter khusus
        text = re.sub(f"[{string.punctuation}]", "", text)
        text = re.sub(r"\d+", "", text)

        # Tokenisasi
        # tokens = word_tokenize(text)

        # Stopword Removal (Bahasa Indonesia)
        # stop_words = set(stopwords.words('indonesian'))
        # filtered_tokens = [word for word in tokens if word not in stop_words]

        # Stemming (Bahasa Indonesia)
        # stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]

        # Gabungkan kembali menjadi teks
        # processed_text = " ".join(stemmed_tokens)

        return text
