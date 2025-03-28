from flask import Blueprint
from src.routes.predict import predict_bp

# Inisialisasi Blueprint utama untuk routes
routes_bp = Blueprint("routes", __name__)

# Register semua blueprint
routes_bp.register_blueprint(predict_bp)
