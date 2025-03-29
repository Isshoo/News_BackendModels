from flask import Blueprint
from src.api.controllers.dataset_controller import DatasetController

dataset_bp = Blueprint("dataset", __name__, url_prefix="/dataset")
dataset_controller = DatasetController()

dataset_bp.route("/upload", methods=["POST"]
                 )(dataset_controller.upload_dataset)
dataset_bp.route("/data", methods=["GET"])(dataset_controller.get_dataset)
dataset_bp.route("/total", methods=["GET"])(dataset_controller.get_total_data)
