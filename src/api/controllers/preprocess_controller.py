import os
from flask import request, jsonify
from src.api.services.preprocess_service import PreprocessService


class PreprocessController:
    def __init__(self):
        self.preprocess_service = PreprocessService()

    def preprocess_dataset(self):
        """ Preprocessing dataset yang sudah diunggah """
        success = self.preprocess_service.preprocess_dataset()
        if not success:
            return jsonify({"error": "No dataset available for preprocessing"}), 400

        return jsonify({"message": "Dataset preprocessed successfully"})

    def get_preprocessed_dataset(self):
        """ Ambil dataset yang sudah diproses """
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 10))

        data = self.preprocess_service.fetch_dataset(
            page, limit, processed=True)
        return jsonify({"data": data})

    def get_total_preprocessed_data(self):
        """ Ambil total data dalam dataset yang sudah diproses """
        total = self.preprocess_service.count_dataset(processed=True)
        return jsonify({"total_data": total})

    def update_label(self):
        """ Mengubah label manual dataset yang sudah diproses """
        data = request.json
        if "index" not in data or "topik" not in data:
            return jsonify({"error": "Invalid request"}), 400

        success = self.preprocess_service.update_label(
            data["index"], data["topik"])
        if not success:
            return jsonify({"error": "Failed to update label"}), 400

        return jsonify({"message": "Label updated successfully"})

    def delete_data(self):
        """ Menghapus baris dataset yang sudah diproses """
        data = request.json
        if "index" not in data:
            return jsonify({"error": "Invalid request"}), 400

        success = self.preprocess_service.delete_data(data["index"])
        if not success:
            return jsonify({"error": "Failed to delete data"}), 400

        return jsonify({"message": "Data deleted successfully"})

    def add_data(self):
        """ Menambahkan data baru ke dataset yang sudah diproses """
        data = request.json
        if "contentSnippet" not in data or "topik" not in data:
            return jsonify({"error": "Invalid request"}), 400

        success = self.preprocess_service.add_data(
            data["contentSnippet"], data["topik"])
        if not success:
            return jsonify({"error": "Failed to add data"}), 400

        return jsonify({"message": "Data added successfully"})

    def get_topics_distribution(self):
        """ Mendapatkan daftar topik dan jumlahnya dalam dataset """
        topics_distribution = self.preprocess_service.get_topics_distribution()
        return jsonify({"topics_distribution": topics_distribution}), 200
