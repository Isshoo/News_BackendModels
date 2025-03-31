import os
from flask import request, jsonify
from src.api.services.dataset_service import DatasetService


class DatasetController:
    def __init__(self):
        self.dataset_service = DatasetService()

    def upload_dataset(self):
        """ Mengunggah dataset, menyimpannya, dan menjalankan preprocessing """
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Cek ekstensi file (harus .csv)
        if not file.filename.lower().endswith('.csv'):
            return jsonify({"error": "Only CSV files are allowed"}), 400

        dataset_name = os.path.splitext(file.filename)[0]

        # Cek apakah dataset dengan nama yang sama sudah ada
        existing_datasets = self.dataset_service.fetch_datasets()
        if any(ds['name'] == dataset_name for ds in existing_datasets):
            return jsonify({"error": "Dataset with the same name already exists"}), 400

        filepath = os.path.join(
            self.dataset_service.DATASET_DIR, file.filename)
        file.save(filepath)

        dataset_info = self.dataset_service.save_dataset(
            filepath, dataset_name)

        return jsonify({
            "message": "Dataset uploaded and processed successfully",
            "dataset": dataset_info
        }), 200

    def get_datasets(self):
        """ Mengambil semua dataset yang tersimpan """
        datasets = self.dataset_service.fetch_datasets()
        return jsonify(datasets), 200

    def get_dataset(self, dataset_id):
        """ Mengambil dataset tertentu dengan paginasi """
        if dataset_id is None:
            return jsonify({"error": "dataset_id is required"}), 400
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 10))

        result = self.dataset_service.fetch_dataset(dataset_id, page, limit)
        if result is None:
            return jsonify({"error": "Dataset not found"}), 404

        return jsonify(result), 200

    def delete_dataset(self, dataset_id):
        """ Menghapus dataset tertentu """
        if dataset_id is None:
            return jsonify({"error": "dataset_id is required"}), 400
        success = self.dataset_service.delete_dataset(dataset_id)
        if not success:
            return jsonify({"error": "Dataset not found"}), 404

        return jsonify({"message": "Dataset deleted successfully"}), 200
