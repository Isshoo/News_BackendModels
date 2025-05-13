from flask import request, jsonify
from src.api.services.predict_service import PredictService
import os


class PredictController:
    predict_service = PredictService()

    def __init__(self):
        pass

    def predict(self):
        """ Menerima input teks dan mengembalikan hasil prediksi """
        try:
            data = request.json
            if not data or "text" not in data:
                return jsonify({"error": "Text is required"}), 400

            text = data["text"]

            if "model_path" not in data:
                result = self.predict_service.predict(text)
                return jsonify(result), 200

            model_path = data["model_path"]

            if model_path == '':
                result = self.predict_service.predict(text)
                return jsonify(result), 200

            result = self.predict_service.predict(text, model_path)

            return jsonify(result), 200

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    def predict_from_csv(self):
        """ Menerima file CSV dan mengembalikan hasil prediksi """
        try:
            if 'file' not in request.files:
                return jsonify({"error": "No file provided"}), 400

            file = request.files['file']

            if file.filename == '':
                return jsonify({"error": "No file selected"}), 400

            if not file.filename.lower().endswith('.csv'):
                return jsonify({"error": "Only CSV files are allowed"}), 400

            dataset_name = os.path.splitext(file.filename)[0].lower()

            file_path = os.path.join(
                self.predict_service.PREDICT_DIR, dataset_name + '.csv')
            file.save(file_path)

            # ==== VALIDASI TAMBAHAN DIMULAI ====
            import pandas as pd

            # Cek apakah file kosong
            try:
                df = pd.read_csv(file_path, sep=',')
            except pd.errors.EmptyDataError:
                return jsonify({"error": "Uploaded CSV file is empty or has no parsable columns"}), 400

            # 1. Cek ukuran file (maks 5MB)
            file.seek(0, os.SEEK_END)
            file_size = file.tell()
            file.seek(0)  # Reset posisi baca
            if file_size > 5 * 1024 * 1024:
                return jsonify({"error": "File size exceeds 5MB limit"}), 400

            # 3. Cek kolom yang wajib ada
            required_columns = {"contentSnippet"}
            if not required_columns.issubset(df.columns):
                return jsonify({"error": "CSV must contain 'contentSnippet' columns"}), 400

            # 4. Cek apakah data kosong
            if df.empty:
                return jsonify({"error": "CSV has no data"}), 400

            # 6. Validasi jumlah total data
            if len(df) < 2:
                return jsonify({"error": "Dataset must contain at least 2 rows"}), 400

            if len(df) > 20:
                return jsonify({"error": "Dataset must contain at most 20 rows"}), 400

            # ==== VALIDASI TAMBAHAN SELESAI ====

            if 'model_path' not in request.form:
                result = self.predict_service.predict_csv(
                    file_path)
                return jsonify(result), 200

            model_path = request.form['model_path']

            if model_path == '':
                result = self.predict_service.predict_csv(file_path)
                return jsonify(result), 200

            result = self.predict_service.predict_csv(file_path, model_path)

            return jsonify(result), 200

        except Exception as e:
            return jsonify({"error": str(e)}), 500
