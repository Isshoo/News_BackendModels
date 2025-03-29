import os
from flask import request, jsonify
from src.api.services.dataset_service import DatasetService


class DatasetController:
    UPLOAD_FOLDER = "src/datasets/"

    def __init__(self):
        self.dataset_service = DatasetService()

    def upload_dataset(self):
        """ Mengunggah dataset, menyimpannya, dan menjalankan preprocessing """
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        filepath = os.path.join(self.UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        processed_file = self.dataset_service.save_dataset(filepath)

        return jsonify({
            "message": "Dataset uploaded and processed successfully",
            "file": processed_file
        }), 200

    def get_dataset(self):
        """ Mengambil dataset dengan paginasi """
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 10))

        data = self.dataset_service.fetch_dataset(page, limit)
        return jsonify({"data": data}), 200

    def get_total_data(self):
        """ Menghitung jumlah total data dalam dataset """
        total = self.dataset_service.count_dataset()
        return jsonify({"total_data": total}), 200
