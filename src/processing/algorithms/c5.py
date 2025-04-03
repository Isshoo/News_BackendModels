import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from tqdm import tqdm  # Import tqdm


class TreeNode:
    def __init__(self, word=None, label=None):
        self.word = word
        self.label = label
        self.children = {}

    def is_leaf(self):
        return self.label is not None


class CustomC5:
    def __init__(self):
        self.root = None

    def compute_entropy(self, labels):
        label_counts = Counter(labels)
        total_samples = len(labels)
        probs = np.array(list(label_counts.values())) / total_samples
        return -np.sum(probs * np.log2(probs)) if total_samples > 0 else 0

    def compute_information_gain(self, word, dataset, labels):
        H_S = self.compute_entropy(labels)
        word_occurrences = [labels[i]
                            for i, text in enumerate(dataset) if word in text.split()]
        H_word = self.compute_entropy(word_occurrences)
        filtered_labels = [labels[i] for i, text in enumerate(
            dataset) if word not in text.split()]
        H_without_word = self.compute_entropy(filtered_labels)
        S_word = sum(1 for text in dataset if word in text.split())
        S_not_word = len(dataset) - S_word
        return H_S - ((S_word / len(dataset)) * H_word + (S_not_word / len(dataset)) * H_without_word)

    def build_tree(self, dataset, labels):
        unique_labels = set(labels)
        if len(unique_labels) == 1:
            return TreeNode(label=labels[0])

        words = set(word for text in dataset for word in text.split())
        if not words:
            return TreeNode(label=Counter(labels).most_common(1)[0][0])

        # Add tqdm for word iteration
        gains = {}
        for word in tqdm(words, desc="Evaluating words for splits", total=len(words)):
            gains[word] = self.compute_information_gain(word, dataset, labels)

        best_word = max(gains, key=gains.get)
        if gains[best_word] == 0:
            return TreeNode(label=Counter(labels).most_common(1)[0][0])

        node = TreeNode(word=best_word)
        dataset_with_word = [
            text for text in dataset if best_word in text.split()]
        labels_with_word = [labels[i] for i, text in enumerate(
            dataset) if best_word in text.split()]
        dataset_without_word = [
            text for text in dataset if best_word not in text.split()]
        labels_without_word = [labels[i] for i, text in enumerate(
            dataset) if best_word not in text.split()]

        node.children["with"] = self.build_tree(
            dataset_with_word, labels_with_word)
        node.children["without"] = self.build_tree(
            dataset_without_word, labels_without_word)
        return node

    def fit(self, X_train, y_train):
        self.root = self.build_tree(X_train, y_train)

    def predict_one(self, text, node):
        if node.is_leaf():
            return node.label
        if node.word in text.split():
            return self.predict_one(text, node.children["with"])
        else:
            return self.predict_one(text, node.children["without"])

    def predict(self, X_test):
        return [self.predict_one(text, self.root) for text in tqdm(X_test, desc="Making Predictions", total=len(X_test))]

    def save_model(self, filepath="src/storage/models/algorithms/decision_tree_model.pkl"):
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(filepath="src/storage/models/algorithms/decision_tree_model.pkl"):
        with open(filepath, "rb") as f:
            return pickle.load(f)


def plot_tree_matplotlib(node, graph=None, parent_name=None):
    if graph is None:
        graph = nx.DiGraph()

    node_name = node.label if node.is_leaf() else node.word
    if parent_name:
        graph.add_edge(parent_name, node_name)

    for key, child in node.children.items():
        plot_tree_matplotlib(child, graph, node_name)

    return graph


def draw_tree(graph):
    pos = nx.spring_layout(graph, seed=42)
    nx.draw(graph, pos, with_labels=True, node_size=3000, node_color='skyblue',
            font_size=10, font_weight='bold', edge_color='gray')
    plt.show()


if __name__ == "__main__":
    dataset_path = "./src/storage/datasets/base/news_dataset_default_preprocessed_stemmed.csv"
    df = pd.read_csv(dataset_path, sep=",", encoding="utf-8")

    if df.empty:
        raise ValueError("Dataset kosong. Cek dataset Anda!")

    X_texts = df["preprocessedContent"].values
    y = df["topik"].values

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_texts, y_encoded, test_size=0.2, stratify=y_encoded, random_state=100
    )

    try:
        c5 = CustomC5.load_model()
        print("Model yang tersimpan berhasil dimuat!")
    except FileNotFoundError:
        print("Model tidak ditemukan, melatih ulang...")
        c5 = CustomC5()
        c5.fit(X_train, y_train)
        c5.save_model()

    print("Mengevaluasi model...")
    predictions = c5.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Akurasi model: {accuracy:.2%}")

    print("Menyimpan visualisasi pohon keputusan...")
    graph = plot_tree_matplotlib(c5.root)
    draw_tree(graph)
