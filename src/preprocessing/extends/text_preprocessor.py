from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import nltk
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from src.preprocessing.preprocessor import Preprocessor
import ssl
import nltk

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


class TextPreprocessor(Preprocessor):

    def __init__(self):
        factory = StemmerFactory()
        self.stemmer = factory.create_stemmer()
        stopword_factory = StopWordRemoverFactory()
        self.stopwords = set(stopword_factory.get_stop_words())
        # nltk.download('punkt')

    def preprocess(self, text):
        print(f"Text Awal: {text}")

        # Case Folding: Ubah semua teks menjadi huruf kecil
        text = text.lower()
        print(f"Case Folding: {text}")

        # Menghapus karakter non-UTF-8
        text = text.encode('utf-8', 'ignore').decode('utf-8')

        # Menghapus tanda baca, angka, dan membersihkan teks
        # Hanya menyisakan huruf, angka, dan tanda hubung (-)
        text = re.sub(r"[^\w\s-]", "", text)
        text = re.sub(r"\d+", "", text)  # Menghapus angka
        text = re.sub(r"\s+", " ", text).strip()  # Menghapus spasi berlebih
        print(f"Cleansing: {text}")

        # Menghapus kata berulang seperti "sering-sering" → "sering"
        text = re.sub(r"\b(\w+)([- ]\1)+\b", r"\1", text)

        print(f"Penghapusan Kata Berulang: {text}")

        # Tokenisasi
        tokens = nltk.word_tokenize(text)
        print(f"Tokenization: {tokens}")

        # Menghapus stopwords dan stemming
        tokens = [self.stemmer.stem(word)
                  for word in tokens if word not in self.stopwords]
        print(f"Stemming: {tokens}")

        # Kembalikan teks yang telah diproses
        return " ".join(tokens)
