import sys
from flask import Flask, request, jsonify
from flask_cors import CORS
from src.utilities.preprocessor import Preprocessor
from src.models.deepseek import DeepSeekClassifier
import joblib


app = Flask(__name__)
CORS(app)

# Load Hybrid Model
sys.path.append('./src')
hybrid_model = joblib.load("./src/models/saved/hybrid_model.joblib")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        if not data or "text" not in data:
            return jsonify({"error": "No text provided"}), 400

        text = data["text"]
        clean_text = Preprocessor.preprocess_text(text)

        # Pastikan model hybrid tersedia
        if not hybrid_model:
            return jsonify({"error": "Hybrid model is not loaded"}), 500

        # Coba prediksi Hybrid Model
        try:
            hybrid_prediction = hybrid_model.predict([clean_text])[0]
        except Exception as e:
            return jsonify({"error": f"Hybrid model prediction failed: {str(e)}"}), 500

        # Coba prediksi DeepSeek
        try:
            deepseek_prediction = DeepSeekClassifier.classify(
                clean_text, use_api=True)
        except Exception as e:
            return jsonify({"error": f"DeepSeek classification failed: {str(e)}"}), 500

        category_map = {
            "ekonomi": "Ekonomi",
            "teknologi": "Teknologi",
            "olahraga": "Olahraga",
            "hiburan": "Hiburan",
            "gayahidup": "Gaya Hidup"
        }

        hybrid_prediction = category_map.get(
            hybrid_prediction, hybrid_prediction)

        return jsonify({
            "text": text,
            "hybrid_prediction": hybrid_prediction,
            "deepseek_prediction": deepseek_prediction
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
