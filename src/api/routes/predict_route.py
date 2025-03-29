from flask import Blueprint
from src.api.controllers.predict_controller import PredictController

predict_bp = Blueprint("predict", __name__, url_prefix="/predict")
predict_controller = PredictController()

predict_bp.route("/", methods=["POST"])(predict_controller.predict)
