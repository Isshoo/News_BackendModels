

def map_hybrid_result(result):
    mapping = {
        "ekonomi": "Ekonomi",
        "teknologi": "Teknologi",
        "olahraga": "Olahraga",
        "hiburan": "Hiburan",
        "gayahidup": "Gaya Hidup"
    }
    return mapping.get(result, result)
