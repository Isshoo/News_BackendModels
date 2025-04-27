import time
import joblib
import pandas as pd
from src.preprocessing.extends.text_preprocessor import TextPreprocessor
from src.models.deepseek import DeepSeekClassifier
from src.utilities.map_classification_result import map_classification_result


class NewsClassifier:
    def __init__(self, hybrid_model_path='./src/storage/models/base/hybrid_model.joblib'):
        """Inisialisasi model hybrid dan text preprocessor"""
        try:
            self.hybrid_model = joblib.load(hybrid_model_path)
        except Exception as e:
            print(f"❌ Gagal memuat Hybrid model: {e}")
            self.hybrid_model = None  # Hindari crash jika model tidak bisa dimuat

        self.text_preprocessor = TextPreprocessor()
        self.valid_categories = {
            "Ekonomi", "Teknologi", "Olahraga", "Hiburan", "GayaHidup"}  # Kategori yang valid

    def classify(self, sample_text, max_retries=2):
        """ Mengklasifikasikan teks berita menggunakan model hybrid dan DeepSeek """
        processed_sample_text = self.text_preprocessor.preprocess(sample_text)
        print(f"Preprocessed Text: {processed_sample_text}")

        # jika hasil preprocess ksosng
        if not processed_sample_text:
            return {
                "Preprocessed_Text": processed_sample_text,
                "Hybrid_C5_KNN": "Unknown",
                "DeepSeek": "Unknown",
                "model_error": f"Failed to preprocess text: '{sample_text}'"
            }

        hybrid_error = ""
        try:
            hasil_model_hybrid, reasons = self.hybrid_model.predict(
                [processed_sample_text])
            hasil_model_hybrid = hasil_model_hybrid[0]
            hasil_model_hybrid = map_classification_result(hasil_model_hybrid)
        except Exception as e:
            print(f"❌ Error pada model Hybrid: {e}")
            hasil_model_hybrid = "Unknown"
            hybrid_error = f"Hybrid Model Error: {e}"

        for attempt in range(max_retries):
            try:
                hasil_deepseek = DeepSeekClassifier.classify(
                    processed_sample_text, use_api=True)
                if hasil_deepseek in self.valid_categories:
                    hasil_deepseek = map_classification_result(hasil_deepseek)
                    if hasil_model_hybrid != "Unknown":
                        return {
                            "Preprocessed_Text": processed_sample_text,
                            "Hybrid_C5_KNN": hasil_model_hybrid,
                            "DeepSeek": hasil_deepseek,
                            "model_error": ""
                        }
                    return {
                        "Preprocessed_Text": processed_sample_text,
                        "Hybrid_C5_KNN": hasil_model_hybrid,
                        "DeepSeek": hasil_deepseek,
                        "model_error": f"{hybrid_error}"
                    }
            except Exception as e:
                print(f"❌ Error pada DeepSeek: {e}")

            print(
                f"🔄 DeepSeek gagal pada percobaan {attempt + 1}. Retrying...")
            time.sleep(1)

        return {
            "Preprocessed_Text": processed_sample_text,
            "Hybrid_C5_KNN": hasil_model_hybrid,
            "DeepSeek": "-",
            "model_error": f"{hybrid_error} & Deepseek Rate Limit Exceeded"
        }

    def classify_csv(self, csv_file_path):
        """ Mengklasifikasikan CSV yang berisi berita """
        try:
            df = pd.read_csv(csv_file_path, encoding="utf-8",
                             on_bad_lines="skip")

            # Pastikan kolom yang diperlukan ada
            required_columns = {"contentSnippet"}
            if not required_columns.issubset(df.columns):
                return {"error": f"File CSV harus memiliki kolom: {', '.join(required_columns)}"}

            df = df.dropna(subset=["contentSnippet"])

            # Inisialisasi kolom hasil
            preprocessed_texts = []
            hybrid_results = []
            deepseek_results = []
            errors = []

            # Looping per baris untuk klasifikasi
            for text in df["contentSnippet"]:
                result = self.classify(text)
                preprocessed_texts.append(result["Preprocessed_Text"])
                hybrid_results.append(result["Hybrid_C5_KNN"])
                deepseek_results.append(result["DeepSeek"])
                errors.append(result["model_error"])

            # Tambahkan hasil ke dataframe
            df["Preprocessed_Text"] = preprocessed_texts
            df["Hybrid_C5_KNN"] = hybrid_results
            df["DeepSeek"] = deepseek_results
            df["model_error"] = errors

            return df.to_dict(orient="records")

        except pd.errors.EmptyDataError:
            return {"error": "File CSV kosong."}
        except pd.errors.ParserError as e:
            return {"error": f"Kesalahan parsing CSV: {str(e)}"}
        except UnicodeDecodeError:
            return {"error": "Encoding tidak valid. Coba simpan file sebagai UTF-8."}
        except Exception as e:
            return {"error": f"Kesalahan internal: {str(e)}"}
