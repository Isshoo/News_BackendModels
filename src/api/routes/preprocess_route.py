from flask import Blueprint
from src.api.controllers.preprocess_controller import PreprocessController

preprocess_bp = Blueprint("preprocess", __name__, url_prefix="/dataset")
preprocess_controller = PreprocessController()

preprocess_bp.route(
    "/preprocess", methods=["POST"])(preprocess_controller.preprocess_dataset)  # Proses dataset
# Ambil dataset hasil preprocessing
preprocess_bp.route(
    "/preprocessed/data", methods=["GET"])(preprocess_controller.get_preprocessed_dataset)
# Total data yang sudah diproses
preprocess_bp.route(
    "/preprocessed/total", methods=["GET"])(preprocess_controller.get_total_preprocessed_data)
# Ubah label manual
preprocess_bp.route(
    "/preprocessed/update", methods=["PUT"])(preprocess_controller.update_label)
# Hapus baris dataset
preprocess_bp.route(
    "/preprocessed/delete", methods=["DELETE"])(preprocess_controller.delete_data)
# Tambah data baru
preprocess_bp.route(
    "/preprocessed/add", methods=["POST"])(preprocess_controller.add_data)
