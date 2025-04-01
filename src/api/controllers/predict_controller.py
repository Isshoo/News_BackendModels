from flask import request, jsonify
from src.api.services.predict_service import PredictService


class PredictController:
    def __init__(self):
        self.predict_service = PredictService()

    def predict(self):
        """ Menerima input teks dan mengembalikan hasil prediksi """
        try:
            data = request.json
            if not data or "text" not in data or "model_path" not in data:
                return jsonify({"error": "Invalid request"}), 400

            text = data["text"]
            model_path = data["model_path"]
            result = self.predict_service.predict(text, model_path)

            return jsonify(result), 200

        except Exception as e:
            return jsonify({"error": str(e)}), 500
