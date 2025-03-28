from flask import Blueprint, request, jsonify
from src.preprocessing.preprocessor import Preprocessor
from src.models.deepseek import DeepSeekClassifier
from src.utilities.map_hybrid_result import map_hybrid_result
import joblib
import sys

predict_bp = Blueprint("predict", __name__)

# Load Hybrid Model
sys.path.append('./src')
hybrid_model = joblib.load("./src/models/saved/hybrid_model.joblib")


@predict_bp.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        if not data or "text" not in data:
            return jsonify({"error": "No text provided"}), 400

        text = data["text"]
        clean_text = Preprocessor.preprocess_text(text)

        if not hybrid_model:
            return jsonify({"error": "Hybrid model is not loaded"}), 500

        try:
            hybrid_prediction = hybrid_model.predict([clean_text])[0]
        except Exception as e:
            return jsonify({"error": f"Hybrid model prediction failed: {str(e)}"}), 500

        try:
            deepseek_prediction = DeepSeekClassifier.classify(
                clean_text, use_api=True)
        except Exception as e:
            return jsonify({"error": f"DeepSeek classification failed: {str(e)}"}), 500

        hybrid_prediction = map_hybrid_result(hybrid_prediction)

        return jsonify({
            "text": text,
            "hybrid_prediction": hybrid_prediction,
            "deepseek_prediction": deepseek_prediction
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
