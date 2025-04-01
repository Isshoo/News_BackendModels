from flask import request, jsonify
from src.api.services.predict_service import PredictService
import os


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

    def predict_from_csv(self):
        """ Menerima file CSV dan mengembalikan hasil prediksi """
        try:
            if 'file' not in request.files or 'model_path' not in request.form:
                return jsonify({"error": "Invalid request"}), 400

            file = request.files['file']
            model_path = request.form['model_path']

            file = request.files['file']
            if file.filename == '':
                return jsonify({"error": "No file selected"}), 400

            # Cek ekstensi file (harus .csv)
            if not file.filename.lower().endswith('.csv'):
                return jsonify({"error": "Only CSV files are allowed"}), 400

            dataset_name = os.path.splitext(file.filename)[0]
            file_path = os.path.join(
                self.predict_service.PREDICT_DIR, dataset_name + '.csv')
            file.save(file_path)

            result = self.predict_service.predict_csv(file_path, model_path)

            return jsonify(result), 200

        except Exception as e:
            return jsonify({"error": str(e)}), 500
