from flask import Blueprint
from src.api.controllers.dataset_controller import upload_dataset, get_dataset, get_total_data

dataset_bp = Blueprint('dataset', __name__)

dataset_bp.route('/datasets/upload', methods=['POST'])(upload_dataset)
dataset_bp.route('/datasets', methods=['GET'])(get_dataset)
dataset_bp.route('/datasets/count', methods=['GET'])(get_total_data)
