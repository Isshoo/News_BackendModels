from src.preprocessing.extends.text_preprocessor import TextPreprocessor
from src.models.deepseek import DeepSeekClassifier
from src.utilities.map_hybrid_result import map_hybrid_result
import joblib
import pandas as pd


class NewsClassifier:
    def __init__(self, hybrid_model_path='./src/storage/models/base/hybrid_model.joblib'):
        self.hybrid_model = joblib.load(hybrid_model_path)
        self.text_preprocessor = TextPreprocessor()

    def classify(self, sample_text):
        """ Mengklasifikasikan teks berita menggunakan model hybrid dan DeepSeek """
        processed_sample_text = self.text_preprocessor.preprocess(sample_text)

        hasil_model_hybrid = self.hybrid_model.predict(
            [processed_sample_text])[0]
        hasil_model_hybrid = map_hybrid_result(hasil_model_hybrid)

        hasil_deepseek = DeepSeekClassifier.classify(
            processed_sample_text, use_api=True)

        return {
            "Preprocessed_Text": processed_sample_text,
            "Hybrid_C5_KNN": hasil_model_hybrid,
            "DeepSeek": hasil_deepseek
        }

    import pandas as pd

    def classify_csv(self, csv_file_path):
        """ Mengklasifikasikan CSV yang berisi teks berita dan label manual """
        try:
            # Membaca file CSV
            df = pd.read_csv(csv_file_path, sep=";", encoding="utf-8")

            # Cek apakah kolom yang diperlukan ada dalam CSV
            required_columns = {"contentSnippet", "topik"}
            missing_columns = required_columns - set(df.columns)
            if missing_columns:
                return {"error": f"Kolom yang hilang: {', '.join(missing_columns)}. Kolom 'contentSnippet' dan 'topik' harus ada dalam file CSV."}

            # Hapus baris dengan teks kosong pada kolom yang dibutuhkan
            df = df.dropna(subset=["contentSnippet", "topik"])

            # Terapkan klasifikasi pada setiap baris di kolom 'contentSnippet'
            df[["Preprocessed_Text", "Hybrid_C5_KNN", "DeepSeek"]] = df["contentSnippet"].apply(
                lambda text: pd.Series(self.classify(text))
            )

            # Kembalikan hasil sebagai list of dicts
            return df.to_dict(orient="records")

        except pd.errors.EmptyDataError:
            return {"error": "File CSV kosong."}
        except pd.errors.ParserError:
            return {"error": "Terjadi kesalahan saat membaca file CSV. Pastikan format file CSV valid."}
        except Exception as e:
            return {"error": f"Kesalahan internal: {str(e)}"}


if __name__ == "__main__":
    import sys
    sys.path.append('./src')

    # Load model hybrid
    hybrid_model_path = './src/models/saved/hybrid_model.joblib'
    classifier = NewsClassifier(hybrid_model_path)
    sample_text = input("Masukkan berita yang akan diklasifikasi: ")
    classifier.classify(sample_text)
