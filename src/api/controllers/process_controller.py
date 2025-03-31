from flask import request, jsonify
from src.api.services.process_service import ProcessService


class ProcessController:
    def __init__(self):
        self.process_service = ProcessService()

    def split_dataset(self):
        try:
            data = request.json
            if "test_size" not in data:
                return jsonify({"error": "Invalid request"}), 400

            test_size = data["test_size"]

            if not isinstance(test_size, (int, float)) or test_size <= 0 or test_size >= 1:
                return jsonify({"error": "Test size must be a positive float between 0 and 1"}), 400

            result = self.process_service.split_dataset(test_size)
            return jsonify(result)

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    def train_model(self):
        try:
            data = request.json
            if "test_size" not in data or "n_neighbors" not in data:
                return jsonify({"error": "Invalid request"}), 400

            n_neighbors = data["n_neighbors"]
            test_size = data["test_size"]

            if not isinstance(n_neighbors, int) or n_neighbors <= 0:
                return jsonify({"error": "Number of neighbors must be a positive integer"}), 400
            if not isinstance(test_size, (int, float)) or test_size <= 0 or test_size >= 1:
                return jsonify({"error": "Test size must be a positive float between 0 and 1"}), 400

            result = self.process_service.train_model(n_neighbors, test_size)
            return jsonify(result)

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    def model_evaluation(self):
        try:
            result = self.process_service.model_evaluation()
            return jsonify(result)

        except Exception as e:
            return jsonify({"error": str(e)}), 500
