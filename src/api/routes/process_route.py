from flask import Blueprint
from src.api.controllers.process_controller import ProcessController

process_bp = Blueprint("process", __name__)
process_controller = ProcessController()

process_bp.route("/process/split",
                 methods=["POST"])(process_controller.split_dataset)
process_bp.route("/process/train",
                 methods=["POST"])(process_controller.train_model)
