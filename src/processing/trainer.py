import pandas as pd
from sklearn.model_selection import train_test_split
from src.models.hybrid import HybridClassifier
from src.utilities.model_evaluations import evaluate_model
from src.utilities.save_model import save_model
from sklearn.preprocessing import LabelEncoder
from src.utilities.map_classification_result import map_classification_result
from itertools import product
import time


class HybridModelTrainer:
    def __init__(self, dataset_path):
        """Inisialisasi trainer dengan dataset dalam bentuk DataFrame"""

        self.df = pd.read_csv(dataset_path, sep=",", encoding="utf-8")
        self.test_df = pd.read_csv(
            "./src/storage/datasets/base/DataTesting2.csv", sep=",", encoding="utf-8")

        if self.df.empty:
            raise ValueError("Dataset kosong. Cek dataset Anda!")

    def train(self, n_neighbors=5, test_size=0.2, c5_threshold=0.75, max_features=None):
        """Melatih model Hybrid C5.0-KNN"""

        X_texts = self.df["preprocessedContent"].values
        y = self.df["topik"].values

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        X_train, X_test, y_train, y_test, raw_train, raw_test = train_test_split(
            X_texts, y_encoded, X_texts, test_size=test_size, stratify=y_encoded, random_state=100
        )

        hybrid_model = HybridClassifier(
            n_neighbors, c5_threshold=c5_threshold, max_features=max_features)

        # hitung waktu latihan
        start_time = time.time()
        # Latih model
        hybrid_model.fit(X_train, y_train, raw_train, le)

        end_time = time.time()

        training_time = end_time - start_time

        word_stats_df = hybrid_model.get_word_stats()

        tfidf_stats = hybrid_model.get_tfidf_word_stats(X_train)

        # Prediksi hasil
        y_pred, y_pred_reason = hybrid_model.predict(X_test)

        # Ambil vektor untuk X_test
        X_test_vectors = hybrid_model.vectorizer.transform(X_test)

        # Hitung tetangga terdekat untuk setiap data uji
        all_neighbors = []
        for i in range(len(X_test)):
            # hanya simpan kalau diprediksi oleh KNN
            if y_pred_reason[i].startswith("KNN"):
                neighbors = hybrid_model.knn.get_neighbors_info(
                    X_test_vectors[i], k=n_neighbors)[0]
                all_neighbors.append({
                    "test_index": i,
                    "test_text": raw_test[i],
                    "predicted_label": le.inverse_transform([y_pred[i]])[0],
                    "true_label": le.inverse_transform([y_test[i]])[0],
                    "reason": y_pred_reason[i],
                    "neighbors": neighbors
                })

        # Konversi ke DataFrame
        rows = []
        for item in all_neighbors:
            for neighbor in item["neighbors"]:
                rows.append({
                    "test_index": item["test_index"],
                    "test_text": item["test_text"],
                    "predicted_label": item["predicted_label"],
                    "true_label": item["true_label"],
                    "reason": item["reason"],
                    "neighbor_index": neighbor["index"],
                    "neighbor_label": map_classification_result(neighbor["label"]),
                    "neighbor_distance": neighbor["distance"],
                    "neighbor_text": neighbor["text"]
                })

        df_neighbors = pd.DataFrame(rows)

        # Buat DataFrame hasil prediksi
        predict_results = []
        for i in range(len(X_test)):
            predict_results.append({
                "index": i,
                "text": raw_test[i],
                "true_label": le.inverse_transform([y_test[i]])[0],
                "predicted_label": le.inverse_transform([y_pred[i]])[0],
                "predict_by": y_pred_reason[i],
                "is_correct": le.inverse_transform([y_test[i]])[0] == le.inverse_transform([y_pred[i]])[0]
            })

        df_predict_results = pd.DataFrame(predict_results)

        # Evaluasi model
        evaluation_results = evaluate_model(y_test, y_pred)

        # evaluasi testing
        X_Testing = self.test_df["preprocessedContent"].values
        Y_Testing = self.test_df["topik"].values

        le_testing = LabelEncoder()
        y_testing = le_testing.fit_transform(Y_Testing)

        y_testing_pred, y_testing_pred_reason = hybrid_model.predict(X_Testing)

        predict_results_testing = []
        for i in range(len(X_Testing)):
            predict_results_testing.append({
                "index": i,
                "text": X_Testing[i],
                "true_label": le.inverse_transform([y_testing[i]])[0],
                "predicted_label": le.inverse_transform([y_testing_pred[i]])[0],
                "predict_by": y_testing_pred_reason[i],
                "is_correct": le.inverse_transform([y_testing[i]])[0] == le.inverse_transform([y_testing_pred[i]])[0]
            })

        df_predict_results_testing = pd.DataFrame(predict_results_testing)

        evaluation_results_testing = evaluate_model(y_testing, y_testing_pred)

        return hybrid_model, evaluation_results, word_stats_df, tfidf_stats, df_neighbors, df_predict_results, df_predict_results_testing, evaluation_results_testing, training_time

    def train_with_gridsearch(self, param_grid=None):
        """Melatih model Hybrid C5.0-KNN dengan Grid Search untuk mencari parameter terbaik"""

        X_texts = self.df["preprocessedContent"].values
        y = self.df["topik"].values

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        # Default grid search parameters
        if param_grid is None:
            param_grid = {
                "n_neighbors": [5, 7, 9, 11],  # Coba beberapa nilai KNN
                # Coba beberapa split train-test
                "test_size": [0.2, 0.25, 0.3],
                # Coba berbagai random state
                "random_state": [42, 100],
                # Coba beberapa split C5.0
                "c5_threshold": [0.75],
                # Coba beberapa nilai maksimum fitur
                "max_features": [None]
            }

        best_score = 0
        best_model = None
        best_params = None
        results = []

        # Loop melalui semua kombinasi parameter
        for n_neighbors, test_size, random_state, c5_threshold, max_features in product(
            param_grid["n_neighbors"], param_grid["test_size"], param_grid["random_state"], param_grid["c5_threshold"], param_grid["max_features"]
        ):
            print(
                f"🔍 Evaluating Hybrid Model with n_neighbors={n_neighbors}, test_size={test_size}, random_state={random_state}, c5_threshold={c5_threshold}")

            print(f"📊 Data size before split: {len(X_texts)}")

            X_train, X_test, y_train, y_test, raw_train, raw_test = train_test_split(
                X_texts, y_encoded, X_texts, test_size=test_size, stratify=y_encoded, random_state=random_state
            )

            print(f"�� Train size: {len(X_train)}, Test size: {len(X_test)}")

            hybrid_model = HybridClassifier(
                n_neighbors=n_neighbors, c5_threshold=c5_threshold, max_features=max_features)
            # Latih model
            start_time = time.time()
            hybrid_model.fit(X_train, y_train, raw_train, le)
            train_duration = time.time() - start_time

            # Prediksi hasil
            y_pred, _ = hybrid_model.predict(X_test)

            # Evaluasi model
            evaluation_results = evaluate_model(y_test, y_pred)
            accuracy = evaluation_results["accuracy"]

            print(f"✅ Accuracy: {accuracy:.4f}")

            results.append({
                "model": hybrid_model,
                "params": {
                    "n_neighbors": n_neighbors,
                    "test_size": test_size,
                    "random_state": random_state,
                    "c5_threshold": c5_threshold,
                    "train_duration": train_duration,
                    "max_features": max_features
                },
                "accuracy": accuracy
            })

            if accuracy > best_score:
                best_score = accuracy
                best_model = hybrid_model
                best_params = {
                    "n_neighbors": n_neighbors,
                    "test_size": test_size,
                    "random_state": random_state,
                    "c5_threshold": c5_threshold,
                    "train_duration": train_duration,
                    "max_features": max_features
                }

        # Urutkan hasil berdasarkan akurasi (tertinggi ke terendah)
        sorted_results = sorted(
            results, key=lambda x: x["accuracy"], reverse=True)

        print("\nRank\tAccuracy\tTest Size\tN_Neighbors\tC5 Threshold\tRandom State\tTrain Duration\tMax Features")
        print(
            "----\t--------\t---------\t----------\t-----------\t------------\t------------\t-------------")
        for i, result in enumerate(sorted_results, 1):
            accuracy = result["accuracy"]
            params = result["params"]
            print(
                f"{i}\t{accuracy:.4f}\t\t{params['test_size']}\t\t{params['n_neighbors']}\t\t{params['c5_threshold']}\t\t{params['random_state']}\t\t{params['train_duration']:.2f}s\t\t{params['max_features']}")

        return best_model, best_params, best_score


if __name__ == "__main__":
    dataset_path = "./src/storage/datasets/preprocessed/raw_news_dataset5_original_preprocessed.csv"
    trainer = HybridModelTrainer(dataset_path)

    # # Training model secara manual
    # trainer.train(n_neighbors=5, test_size=0.25,
    #               c5_threshold=0.75, max_features=None)

    # Training model dengan Grid Search
    best_model, best_params, best_score = trainer.train_with_gridsearch()
    print(f"\nParameter terbaik ditemukan: {best_params}")
    print(f"Akurasi terbaik: {best_score:.4f}")
    isSimpan = input("Apakah model akan disimpan sebagai default? y/n: ")
    if isSimpan.lower() == "y":
        save_model(best_model, "./src/storage/models/base/hybrid_model.joblib")
        print("Model hybrid disimpan sebagai default")
    else:
        print("Training Selesai")
