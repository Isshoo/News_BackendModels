import re
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from src.preprocessing.preprocessor import Preprocessor


class TextPreprocessor(Preprocessor):
    def __init__(self):
        self.stop_words = set(stopwords.words('indonesian'))

        # Stemming dengan Sastrawi
        factory = StemmerFactory()
        self.stemmer = factory.create_stemmer()

    def preprocess(self, texts):
        """ Preprocessing teks """
        preprocessed_texts = []
        for text in texts:
            print(f"\nOriginal text: {text}")  # Debugging

            # Case folding
            text = text.lower()
            print(f"After case folding: {text}")

            # Text cleansing (mengganti simbol dengan spasi)
            text = re.sub(f"[{string.punctuation}]", " ", text)
            text = re.sub(r"\d+", "", text)

            if not text.strip():
                print("Warning: Teks kosong setelah text cleansing.")
                preprocessed_texts.append("")
                continue
            print(f"After cleansing: {text}")

            # Tokenization menggunakan spaCy (lebih akurat)
            tokens = re.split(r"\s+", text.strip())

            print(f"After tokenization: {tokens}")

            if not tokens:
                print("Warning: Token kosong setelah tokenisasi.")
                preprocessed_texts.append("")
                continue

            # Stopword removal
            tokens = [word for word in tokens if word not in self.stop_words]
            print(f"After stopword removal: {tokens}")

            if not tokens:
                print("Warning: Semua token dihapus oleh stopword removal.")
                preprocessed_texts.append("")
                continue

            # Stemming menggunakan Sastrawi
            stemmed_tokens = [self.stemmer.stem(word) for word in tokens]
            print(f"After stemming: {stemmed_tokens}")

            if not stemmed_tokens or all(not tok.strip() for tok in stemmed_tokens):
                print("Warning: Teks kosong setelah stemming.")
                preprocessed_texts.append("")
                continue

            preprocessed_texts.append(" ".join(stemmed_tokens))

        return preprocessed_texts
