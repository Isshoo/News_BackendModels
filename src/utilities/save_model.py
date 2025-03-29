import joblib


def save_model(model):
    """Menyimpan model dalam format joblib"""
    model_path = './src/models/saved/hybrid_model.joblib'
    joblib.dump(model, model_path)
    print(f"Model berhasil disimpan di {model_path}")
