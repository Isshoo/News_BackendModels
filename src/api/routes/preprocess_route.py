# src/api/routes/preprocess_routes.py
from flask import Blueprint
from src.api.controllers.preprocess_controller import PreprocessController

preprocess_bp = Blueprint("preprocess", __name__, url_prefix="/dataset")
preprocess_controller = PreprocessController()

# Route untuk melakukan preprocessing dataset
preprocess_bp.route("/<raw_dataset_id>/preprocess", methods=["POST"]
                    )(preprocess_controller.preprocess_dataset)

# Route untuk membuat salinan dataset yang sudah diproses
preprocess_bp.route("/<raw_dataset_id>/preprocessed/copy", methods=["POST"]
                    )(preprocess_controller.create_preprocessed_copy)

# Route untuk mengambil dataset yang sudah diproses
preprocess_bp.route("/preprocesseds/list", methods=["GET"]
                    )(preprocess_controller.fetch_all_preprocessed_datasets)

# Route untuk mengambil dataset yang sudah diproses
preprocess_bp.route("/<raw_dataset_id>/preprocessed/list", methods=["GET"]
                    )(preprocess_controller.fetch_preprocessed_datasets)

# Route untuk mengambil dataset yang sudah diproses tertentu
preprocess_bp.route("/preprocessed/<dataset_id>", methods=["GET"]
                    )(preprocess_controller.fetch_preprocessed_dataset)

# Route untuk menghapus dataset yang sudah diproses tertentu
preprocess_bp.route("/preprocessed/<dataset_id>", methods=["DELETE"]
                    )(preprocess_controller.delete_preprocessed_dataset)

# Route untuk mengubah label manual dataset yang sudah diproses
preprocess_bp.route("/preprocessed/<dataset_id>/data", methods=["PUT"]
                    )(preprocess_controller.update_data)

# Route untuk menambah data pada dataset yang sudah diproses
preprocess_bp.route("/preprocessed/<dataset_id>/data", methods=["POST"]
                    )(preprocess_controller.add_data)

# Route untuk menghapus baris dataset yang sudah diproses
preprocess_bp.route("/preprocessed/<dataset_id>/data", methods=["DELETE"]
                    )(preprocess_controller.delete_data)

# Route untuk menambah data baru
preprocess_bp.route(
    "/preprocessed/data", methods=["POST"])(preprocess_controller.add_new_data)

# Route untuk melakukan preprocessing data baru
preprocess_bp.route(
    "/preprocessed/process", methods=["POST"])(preprocess_controller.preprocess_new_data)

# Route untuk mengambil data dengan filter
preprocess_bp.route(
    "/preprocessed/data", methods=["GET"])(preprocess_controller.get_preprocessed_data)

# Route untuk mengedit data baru
preprocess_bp.route("/preprocessed/data/<int:index>",
                    methods=["PUT"])(preprocess_controller.edit_new_data)

# Route untuk menghapus data baru
preprocess_bp.route(
    "/preprocessed/data", methods=["DELETE"])(preprocess_controller.delete_new_data)

# Route untuk menandai data sebagai sudah di-train
preprocess_bp.route(
    "/preprocessed/mark-trained", methods=["POST"])(preprocess_controller.mark_as_trained)
