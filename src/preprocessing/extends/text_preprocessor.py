import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from src.preprocessing.preprocessor import Preprocessor


class TextPreprocessor(Preprocessor):
    def __init__(self):
        factory = StemmerFactory()
        self.stemmer = factory.create_stemmer()
        self.stop_words = set(stopwords.words('indonesian'))

    def preprocess(self, texts):
        """ Preprocessing teks """
        preprocessed_texts = []
        for text in texts:
            # Case folding
            text = text.lower()

            # text cleansing
            text = re.sub(f"[{string.punctuation}]", "", text)
            text = re.sub(r"\d+", "", text)

            # Tokenization
            tokens = word_tokenize(text)

            # # Stopword removal
            # tokens = [word for word in tokens if word not in self.stop_words]

            # # Stemming
            # tokens = [self.stemmer.stem(word) for word in tokens]

            preprocessed_texts.append(' '.join(tokens))

        return preprocessed_texts
