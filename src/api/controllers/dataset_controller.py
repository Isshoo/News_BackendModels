import os
from flask import request, jsonify
from src.api.services.dataset_service import DatasetService
from src.api.services.preprocess_service import PreprocessService
from src.api.services.process_service import ProcessService


class DatasetController:
    def __init__(self):
        self.dataset_service = DatasetService()
        self.preprocess_service = PreprocessService()
        self.process_service = ProcessService()

    def upload_dataset(self):
        """ Mengunggah dataset, menyimpannya, dan menjalankan preprocessing """
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Cek ekstensi file (harus .csv)
        if not file.filename.lower().endswith('.csv'):
            return jsonify({"error": "Only CSV files are allowed"}), 400

        dataset_name = os.path.splitext(file.filename)[0].lower()

        # Cek apakah dataset dengan nama yang sama sudah ada
        existing_datasets = self.dataset_service.fetch_datasets()
        if any(ds['name'] == dataset_name for ds in existing_datasets):
            return jsonify({"error": "Dataset with the same name already exists"}), 400

        filepath = os.path.join(
            self.dataset_service.DATASET_DIR, file.filename)
        file.save(filepath)

        # ==== VALIDASI TAMBAHAN DIMULAI ====
        import pandas as pd

        # Cek apakah file kosong
        try:
            df = pd.read_csv(filepath, sep=',')
        except pd.errors.EmptyDataError:
            return jsonify({"error": "Uploaded CSV file is empty or has no parsable columns"}), 400

        # 1. Cek ukuran file (maks 5MB)
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)  # Reset posisi baca
        if file_size > 5 * 1024 * 1024:
            return jsonify({"error": "File size exceeds 5MB limit"}), 400

        # 3. Cek kolom yang wajib ada
        required_columns = {"topik", "contentSnippet"}
        if not required_columns.issubset(df.columns):
            return jsonify({"error": "CSV must contain 'topik' and 'contentSnippet' columns"}), 400

        # 4. Cek apakah data kosong
        if df.empty:
            return jsonify({"error": "CSV has no data"}), 400

        # 5. Validasi topik
        allowed_topics = {"ekonomi", "olahraga",
                          "gayahidup", "teknologi", "hiburan"}
        actual_topics = set(df["topik"].unique())
        invalid_topics = actual_topics - allowed_topics
        if invalid_topics:
            return jsonify({"error": f"Invalid topics found: {', '.join(invalid_topics)}"}), 400

        # Cek apakah kelima topik wajib ada
        missing_required_topics = allowed_topics - actual_topics
        if missing_required_topics:
            return jsonify({"error": f"Dataset must include all 5 required topics, missing: {', '.join(missing_required_topics)}"}), 400

        # 6. Validasi jumlah total data
        if len(df) < 1000:
            return jsonify({"error": "Dataset must contain at least 1000 rows"}), 400

        # 7. Validasi minimal 20 data per topik
        topic_counts = df["topik"].value_counts()
        insufficient_topics = [
            topic for topic in allowed_topics if topic_counts.get(topic, 0) < 200
        ]
        if insufficient_topics:
            return jsonify({
                "error": f"Each topic must have at least 200 data entries. Lacking: {', '.join(insufficient_topics)}"
            }), 400

        # ==== VALIDASI TAMBAHAN SELESAI ====

        dataset_info = self.dataset_service.save_dataset(
            filepath, dataset_name)

        return jsonify({
            "message": "Dataset uploaded and processed successfully",
            "dataset": dataset_info
        }), 200

    def get_datasets(self):
        """ Mengambil semua dataset yang tersimpan """
        datasets = self.dataset_service.fetch_datasets()
        return jsonify(datasets), 200

    def get_dataset(self, dataset_id):
        """ Mengambil dataset tertentu dengan paginasi """
        if dataset_id is None:
            return jsonify({"error": "dataset_id is required"}), 400
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 10))

        result = self.dataset_service.fetch_dataset(dataset_id, page, limit)
        if result is None:
            return jsonify({"error": "Dataset not found"}), 404

        return jsonify(result), 200

    def delete_dataset(self, dataset_id):
        """ Menghapus dataset tertentu """
        if dataset_id is None:
            return jsonify({"error": "dataset_id is required"}), 400

        # cek apakah dataset ada
        if not self.dataset_service.fetch_dataset(dataset_id):
            return jsonify({"error": "Dataset not found"}), 404

        # jika id sama dengan dataset default maka tidak bisa dihapus
        if dataset_id == "default-stemming":
            return jsonify({"error": "Cannot delete default dataset"}), 400

        success = self.dataset_service.delete_dataset(dataset_id)
        if not success:
            return jsonify({"error": "Dataset not found"}), 404

        raw_dataset_id = dataset_id
        # menghapus semua preprocessed datasets dari dataset ini
        preprocessed_datasets = self.preprocess_service.fetch_preprocessed_datasets(
            dataset_id)
        for preprocessed_dataset in preprocessed_datasets:
            resultPre = self.preprocess_service.delete_preprocessed_dataset(
                preprocessed_dataset["id"], raw_dataset_id)
            if resultPre == False:
                return jsonify({"error": "Default preprocessed dataset cannot be deleted"}), 404

        # menghapus semua models dari dataset ini
        models = self.process_service.get_models()
        for model in models:
            if model["raw_dataset_id"] == raw_dataset_id:
                resultMod = self.process_service.delete_model(model["id"])
                if resultMod == False:
                    return jsonify({"error": "Default model cannot be deleted"}), 404

        return jsonify({"message": "Dataset deleted successfully"}), 200
