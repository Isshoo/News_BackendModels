from flask import Blueprint
from src.api.routes.predict_route import predict_bp

# Inisialisasi Blueprint utama untuk routes
routes_bp = Blueprint("routes", __name__)

# Register semua blueprint
routes_bp.register_blueprint(predict_bp)
