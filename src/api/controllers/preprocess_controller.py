# src/api/controllers/preprocess_controller.py
import os
from flask import request, jsonify
from src.api.services.preprocess_service import PreprocessService
from src.api.services.dataset_service import DatasetService
from src.api.services.process_service import ProcessService


class PreprocessController:
    preprocess_service = PreprocessService()
    dataset_service = DatasetService()
    process_service = ProcessService()

    def __init__(self):
        pass

    def preprocess_dataset(self, raw_dataset_id):
        """ Preprocessing dataset yang sudah diunggah """
        if not raw_dataset_id:
            return jsonify({"error": "raw_dataset_id is required"}), 400

        datasets = self.dataset_service.fetch_datasets()

        raw_dataset_path = next(
            (d["path"] for d in datasets if d["id"] == raw_dataset_id), None)
        if not raw_dataset_path:
            return jsonify({"error": "Raw dataset not found"}), 404
        raw_dataset_name = next(
            (d["name"] for d in datasets if d["id"] == raw_dataset_id), None)

        result = self.preprocess_service.preprocess_dataset(
            raw_dataset_id, raw_dataset_path, raw_dataset_name
        )
        if not result:
            return jsonify({"error": "Dataset preprocessing failed"}), 400

        return jsonify({"message": "Dataset preprocessed successfully", "data": result})

    def create_preprocessed_copy(self, raw_dataset_id):
        """ Membuat salinan dataset yang sudah diproses """
        if not raw_dataset_id:
            return jsonify({"error": "raw_dataset_id is required"}), 400
        data = request.json
        if "name" not in data:
            return jsonify({"error": "Invalid request"}), 400

        name = data["name"]
        preprocessed_datasets = self.preprocess_service.fetch_preprocessed_datasets(
            raw_dataset_id)
        if any(d["name"] == name for d in preprocessed_datasets):
            return jsonify({"error": "Preprocessed copy name already exists"}), 400

        result = self.preprocess_service.create_preprocessed_copy(
            raw_dataset_id, name)
        if not result:
            return jsonify({"error": "Failed to create preprocessed copy"}), 400

        return jsonify({"message": "Preprocessed copy created successfully", "data": result})

    def fetch_all_preprocessed_datasets(self):
        """ Ambil semua dataset yang sudah diproses """

        result = self.preprocess_service.fetch_all_preprocessed_datasets()
        return jsonify(result), 200

    def fetch_preprocessed_datasets(self, raw_dataset_id):
        """ Ambil dataset yang sudah diproses """

        if not raw_dataset_id:
            return jsonify({"error": "raw_dataset_id is required"}), 400

        result = self.preprocess_service.fetch_preprocessed_datasets(
            raw_dataset_id)
        return jsonify(result)

    def fetch_preprocessed_dataset(self, dataset_id):
        """ Mengambil dataset yang sudah diproses tertentu """

        if dataset_id is None:
            return jsonify({"error": "dataset_id is required"}), 400
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 10))

        result = self.preprocess_service.fetch_preprocessed_dataset(
            dataset_id, page, limit)
        if not result:
            return jsonify({"error": "Dataset not found"}), 404

        return jsonify(result)

    def delete_preprocessed_dataset(self, dataset_id):
        """ Menghapus dataset yang sudah diproses tertentu """

        if dataset_id is None:
            return jsonify({"error": "dataset_id is required"}), 400

        if not self.preprocess_service.fetch_preprocessed_dataset(dataset_id):
            return jsonify({"error": "Dataset not found"}), 404

        # jika id sama dengan dataset default maka tidak bisa dihapus
        if dataset_id == "default-stemming":
            return jsonify({"error": "Cannot delete default preprocessed dataset"}), 400

        result, status_code = self.preprocess_service.delete_preprocessed_dataset(
            dataset_id)

        if status_code != 200:
            return jsonify(result), status_code

        # delete models for this preprocessed dataset
        models = self.process_service.get_models()
        for model in models:
            if model["preprocessed_dataset_id"] == dataset_id:
                resultMod = self.process_service.delete_model(model["id"])
                if resultMod == False:
                    return jsonify({"error": "Default model cannot be deleted"}), 404

        return jsonify(result), status_code

    def update_data(self, dataset_id):
        """ Mengubah label manual dataset yang sudah diproses """

        if dataset_id is None:
            return jsonify({"error": "dataset_id is required"}), 400
        data = request.json
        if "index" not in data or "topik" not in data or "preprocessedContent" not in data:
            return jsonify({"error": "Invalid request"}), 400

        result, status_code = self.preprocess_service.update_data(
            dataset_id, data["index"], data["topik"], data["preprocessedContent"]
        )
        return jsonify(result), status_code

    def delete_data(self, dataset_id):
        """ Menghapus baris dataset yang sudah diproses """

        if dataset_id is None:
            return jsonify({"error": "dataset_id is required"}), 400
        data = request.json
        if "index" not in data:
            return jsonify({"error": "Invalid request"}), 400

        result, status_code = self.preprocess_service.delete_data(
            dataset_id, data["index"])
        return jsonify(result), status_code

    def add_data(self, dataset_id):
        """ Menambahkan data baru ke dataset yang sudah diproses """
        if dataset_id is None:
            return jsonify({"error": "dataset_id is required"}), 400
        data = request.json
        if "contentSnippet" not in data or "topik" not in data:
            return jsonify({"error": "Invalid request"}), 400

        result, status_code = self.preprocess_service.add_data(
            dataset_id, data["contentSnippet"], data["topik"]
        )
        return jsonify(result), status_code

    def add_new_data(self):
        """Menambahkan data baru ke default preprocessed dataset"""
        data = request.json
        if not isinstance(data, list):
            return jsonify({"error": "Expected an array of data"}), 400

        # Validate each item
        for item in data:
            if "contentSnippet" not in item or "topik" not in item:
                return jsonify({"error": "Each item must contain 'contentSnippet' and 'topik'"}), 400

        result, status_code = self.preprocess_service.add_new_data(data)
        return jsonify(result), status_code

    def preprocess_new_data(self):
        """Melakukan preprocessing pada data baru"""
        result, status_code = self.preprocess_service.preprocess_new_data()
        return jsonify(result), status_code

    def get_preprocessed_data(self):
        """Mengambil data preprocessed dengan filter"""
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 10))
        filter_type = request.args.get('filter', 'all')

        result = self.preprocess_service.fetch_preprocessed_data(
            page, limit, filter_type)
        if "error" in result:
            return jsonify(result), 404
        return jsonify(result), 200

    def edit_new_data(self, index):
        """Mengedit data baru"""
        data = request.json
        changes = {}
        if "topik" in data:
            changes["new_label"] = data["topik"]
        if "contentSnippet" in data:
            changes["new_content"] = data["contentSnippet"]

        if not changes:
            return jsonify({"error": "No changes provided"}), 400

        result, status_code = self.preprocess_service.edit_new_data(
            index, **changes)
        return jsonify(result), status_code

    def delete_new_data(self):
        """Menghapus data baru"""
        data = request.json
        if not isinstance(data, list):
            return jsonify({"error": "Expected an array of indices"}), 400

        result, status_code = self.preprocess_service.delete_new_data(data)
        return jsonify(result), status_code

    def mark_as_trained(self):
        """Menandai data sebagai sudah di-train"""
        data = request.json
        if not isinstance(data, list):
            return jsonify({"error": "Expected an array of indices"}), 400

        result, status_code = self.preprocess_service.mark_data_as_trained(
            data)
        return jsonify(result), status_code
