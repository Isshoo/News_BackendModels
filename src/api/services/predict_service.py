import joblib
from src.preprocessing.extends.text_preprocessor import TextPreprocessor
from src.models.deepseek import DeepSeekClassifier
from src.utilities.map_hybrid_result import map_hybrid_result


class PredictService:
    def __init__(self):
        self.hybrid_model = joblib.load(
            "./src/models/saved/hybrid_model.joblib")
        self.text_preprocessor = TextPreprocessor()

    def predict(self, text):
        """ Mengklasifikasikan teks berita menggunakan model hybrid dan DeepSeek """
        clean_text = self.text_preprocessor.preprocess(text)

        # Prediksi menggunakan Hybrid Model
        try:
            hybrid_prediction = self.hybrid_model.predict([clean_text])[0]
            hybrid_prediction = map_hybrid_result(hybrid_prediction)
        except Exception as e:
            hybrid_prediction = f"Hybrid model error: {str(e)}"

        # Prediksi menggunakan DeepSeek
        try:
            deepseek_prediction = DeepSeekClassifier.classify(
                clean_text, use_api=True)
        except Exception as e:
            deepseek_prediction = f"DeepSeek error: {str(e)}"

        return {
            "text": text,
            "hybrid_prediction": hybrid_prediction,
            "deepseek_prediction": deepseek_prediction
        }
