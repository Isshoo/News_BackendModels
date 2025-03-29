from flask import request, jsonify
from src.api.services.predict_service import PredictService


class PredictController:
    def __init__(self):
        self.predict_service = PredictService()

    def predict(self):
        """ Menerima input teks dan mengembalikan hasil prediksi """
        try:
            data = request.json
            if not data or "text" not in data:
                return jsonify({"error": "No text provided"}), 400

            text = data["text"]
            result = self.predict_service.predict(text)

            return jsonify(result), 200

        except Exception as e:
            return jsonify({"error": str(e)}), 500
