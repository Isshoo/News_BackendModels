import sys
from flask import Flask, request, jsonify
from src.utilities.preprocessor import Preprocessor
from src.models.deepseek import DeepSeekClassifier
import joblib

app = Flask(__name__)


# Load Hybrid Model
sys.path.append('./src')
hybrid_model = joblib.load("./src/models/saved/hybrid_model.joblib")


@app.route("/predict/", methods=["POST"])
def predict():
    data = request.json
    if "text" not in data:
        return jsonify({"error": "No text provided"})

    text = data["text"]
    clean_text = Preprocessor.preprocess_text(text)
    hybrid_prediction = hybrid_model.predict([clean_text])[0]
    deepseek_prediction = DeepSeekClassifier.classify(clean_text, use_api=True)

    category_map = {
        "ekonomi": "Ekonomi",
        "teknologi": "Teknologi",
        "olahraga": "Olahraga",
        "hiburan": "Hiburan",
        "gayahidup": "Gaya Hidup"
    }

    hybrid_prediction = category_map.get(hybrid_prediction, hybrid_prediction)

    return jsonify({
        "text": text,
        "hybrid_prediction": hybrid_prediction,
        "deepseek_prediction": deepseek_prediction
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
