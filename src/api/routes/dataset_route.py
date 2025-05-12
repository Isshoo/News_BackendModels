# src/api/routes/dataset_routes.py
from flask import Blueprint
from src.api.controllers.dataset_controller import DatasetController

dataset_bp = Blueprint("dataset", __name__, url_prefix="/dataset")
dataset_controller = DatasetController()

# Route untuk mengunggah dataset
dataset_bp.route("/upload", methods=["POST"]
                 )(dataset_controller.upload_dataset)

# Route untuk mengambil semua dataset
dataset_bp.route("/list", methods=["GET"])(dataset_controller.get_datasets)

# Route untuk mengambil dataset tertentu dengan paginasi
dataset_bp.route(
    "/<dataset_id>", methods=["GET"])(dataset_controller.get_dataset)

# Route untuk menghapus dataset tertentu
dataset_bp.route(
    "/<dataset_id>", methods=["DELETE"])(dataset_controller.delete_dataset)

# Route untuk menambah data pada dataset
dataset_bp.route("/<dataset_id>/data", methods=["POST"]
                 )(dataset_controller.add_data)

# Route untuk menghapus data dari dataset
dataset_bp.route("/<dataset_id>/data", methods=["DELETE"]
                 )(dataset_controller.delete_data)

# Route untuk melihat riwayat perubahan
dataset_bp.route("/history", methods=["GET"])(dataset_controller.get_history)
dataset_bp.route("/<dataset_id>/history", methods=["GET"]
                 )(dataset_controller.get_history)
