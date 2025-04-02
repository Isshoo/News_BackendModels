

def map_classification_result(result):
    mapping = {
        "ekonomi": "Ekonomi",
        "teknologi": "Teknologi",
        "olahraga": "Olahraga",
        "hiburan": "Hiburan",
        "gayahidup": "Gaya Hidup",
        "GayaHidup": "Gaya Hidup"
    }
    return mapping.get(result, result)
