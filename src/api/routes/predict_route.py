from flask import Blueprint
from src.api.controllers.predict_controller import predict

predict_bp = Blueprint("predict", __name__)

predict_bp.route("/predict", methods=["POST"])(predict)
