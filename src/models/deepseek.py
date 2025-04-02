import ollama
from openai import OpenAI
import re
import os
from dotenv import load_dotenv

load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
VALID_CATEGORIES = {"Ekonomi", "Teknologi", "Olahraga", "Hiburan", "GayaHidup"}


class DeepSeekClassifier:

    @staticmethod
    def classify(text, use_api=False):
        prompt = f"Berdasarkan teks berita berbahasa indonesia berikut: {text}, kira-kira kategori apa yang paling sesuai untuk berita tersebut. pilih di antara Ekonomi, Teknologi, Olahraga, Hiburan, atau GayaHidup. Berikan jawaban hanya berupa satu kata kategori dari antara 5 kategori tersebut tanpa perlu diberikan penjelasan tambahan."

        return (
            DeepSeekClassifier._classify_api(prompt) if use_api
            else DeepSeekClassifier._classify_local(prompt)
        )

    @staticmethod
    def _classify_local(prompt):
        """ Menggunakan model lokal Ollama """
        response = ollama.chat(model='deepseek-r1:1.5b',
                               messages=[{"role": "user", "content": prompt}])

        response_text = response["choices"][0]["message"]["content"].strip()
        response_text = re.sub(r"</?think>", "", response_text).strip()
        return response_text if response_text in VALID_CATEGORIES else "Unknown"

    @staticmethod
    def _classify_api(prompt):
        """ Menggunakan OpenRouter API untuk klasifikasi berita """
        api_key = DEEPSEEK_API_KEY
        if not api_key:
            raise ValueError(
                "DeepSeek API Key tidak ditemukan. Pastikan sudah ada di .env")

        client = OpenAI(
            base_url="https://openrouter.ai/api/v1", api_key=api_key)

        for _ in range(3):  # Maksimal 5 percobaan jika hasil tidak valid
            response = client.chat.completions.create(
                model="deepseek/deepseek-r1-distill-llama-70b:free",
                messages=[{"role": "user", "content": prompt}],
            )
            if hasattr(response, "error"):
                print(f"❌ Error dari API: {response.error}")
                return "Error: Rate Limit Tercapai"

            response_text = response.choices[0].message.content.strip()
            response_text = re.sub(r"</?think>", "", response_text).strip()

            # Pastikan hanya mengembalikan kategori yang valid
            if response_text in VALID_CATEGORIES:
                return response_text

        return "Unknown"


if __name__ == "__main__":
    text = input("Masukkan teks berita: ")

    print("Hasil Klasifikasi: ", DeepSeekClassifier.classify(text, use_api=True))
