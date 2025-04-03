

def map_classification_result(result):
    mapping = {
        0: "Ekonomi",
        1: "Gaya Hidup",
        2: "Hiburan",
        3: "Olahraga",
        4: "Teknologi",
        "ekonomi": "Ekonomi",
        "gayahidup": "Gaya Hidup",
        "hiburan": "Hiburan",
        "olahraga": "Olahraga",
        "teknologi": "Teknologi",
        "GayaHidup": "Gaya Hidup"
    }
    return mapping.get(result, result)
