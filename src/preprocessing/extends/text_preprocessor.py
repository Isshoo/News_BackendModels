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
        text = text.lower()
        print(f"Case Folding: {text}")
        text = re.sub(r"[^a-zA-Z0-9 ]", "", text)
        print(f"Cleansing: {text}")
        tokens = nltk.word_tokenize(text)
        print(f"Tokenization: {tokens}")
        tokens = [self.stemmer.stem(word)
                  for word in tokens if word not in self.stopwords]
        print(f"Stemming: {tokens}")
        return " ".join(tokens)
