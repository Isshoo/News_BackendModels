from flask import Flask
from flask_cors import CORS
from src.api.routes import routes_bp  # Import blueprint utama

app = Flask(__name__)
CORS(app)

# Daftarkan blueprint
app.register_blueprint(routes_bp)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
