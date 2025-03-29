import os
from flask import request, jsonify
from src.api.services.dataset_service import save_dataset, fetch_dataset, count_dataset

UPLOAD_FOLDER = "src/datasets/"


def upload_dataset():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    processed_file = save_dataset(filepath)

    return jsonify({"message": "Dataset uploaded and processed successfully", "file": processed_file})


def get_dataset():
    page = int(request.args.get('page', 1))
    limit = int(request.args.get('limit', 10))

    data = fetch_dataset(page, limit)
    return jsonify({"data": data})


def get_total_data():
    total = count_dataset()
    return jsonify({"total_data": total})
