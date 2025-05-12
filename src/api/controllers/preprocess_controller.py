# src/api/controllers/preprocess_controller.py
from flask import request, jsonify
from src.api.services.preprocess_service import PreprocessService


class PreprocessController:
    def __init__(self):
        self.preprocess_service = PreprocessService()

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
