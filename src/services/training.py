
import joblib
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from src.models.hybrid import HybridClassifier
from src.utilities.preprocessor import Preprocessor
import numpy as np
import pandas as pd


def train_hybrid_model(df):

    X_texts = df["clean_text"].tolist()
    y = df["topik"]

    X_train, X_test, y_train, y_test = train_test_split(
        X_texts, y, test_size=0.2, stratify=y, random_state=42)

    hybrid_model = HybridClassifier()

    # Latih model
    hybrid_model.fit(X_train, y_train)

    # Prediksi hasil
    y_pred = hybrid_model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
    print("\nConfusion Matrix model Hybrid C5.0-KNN:\n", cm)
    print("\nClassification Report model Hybrid C5.0-KNN:\n",
          classification_report(y_test, y_pred))

    # Menyimpan model ke dalam file
    joblib.dump(hybrid_model, './src/models/saved/hybrid_model.joblib')

    # Menyimpan model ke dalam file
    with open('./src/models/saved/hybrid_model.pkl', 'wb') as file:
        pickle.dump(hybrid_model, file)

    return hybrid_model


if __name__ == "__main__":
    # Load dataset
    df = Preprocessor.preprocess_dataset(
        "./src/dataset/dataset-berita-ppl.csv")
    # Latih model dan dapatkan hasilnya
    hybrid_model = train_hybrid_model(df)
