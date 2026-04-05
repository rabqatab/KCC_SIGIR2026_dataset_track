"""
Train Word2Vec and FastText embeddings on Korean legal text.

Uses all available text from the KCC dataset (notes, abstracts, and full case texts).  
The paper states: "nouns and their embeddings extracted from case notes were used as input features" for non-PLM baselines (1D-CNN, LSTM).
"""

import json
import os
from pathlib import Path

from gensim.models import FastText, Word2Vec
from kiwipiepy import Kiwi

EMBED_DIM = 128
WINDOW = 5
MIN_COUNT = 2
EPOCHS = 30
MODEL_DIR = "results/embeddings"

_kiwi = Kiwi()


def extract_nouns(text: str) -> list[str]:
    """Extract nouns from Korean text using Kiwi morphological analyser."""
    return [t.form for t in _kiwi.tokenize(text) if t.tag.startswith("NN")]


def collect_all_sentences(data_dir: str = "dataset") -> list[list[str]]:
    """Collect noun-tokenized sentences from all texts in the dataset.

    Each unique text (note, abstract, or full case) is split on sentence boundaries, 
    tokenized into nouns, and returned as a list of token lists suitable for gensim training.
    """
    unique_texts: set[str] = set()

    for filename in sorted(os.listdir(data_dir)):
        if not filename.endswith(".json"):
            continue
        filepath = os.path.join(data_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        for entry in data.values():
            for field in [
                "query_precedentNote",
                "query_precedentAbstract",
                "query_precedentText",
                "candidate_precedentNote",
                "candidate_precedentAbstract",
                "candidate_precedentText",
            ]:
                text = entry.get(field) or ""
                if text:
                    unique_texts.add(text)

    print(f"Collected {len(unique_texts)} unique texts", flush=True)

    all_sentences: list[list[str]] = []
    done = 0

    for text in unique_texts:
        chunks = [c.strip() for c in text.split(".") if c.strip()]
        if not chunks:
            chunks = [text]
        for chunk in chunks:
            nouns = extract_nouns(chunk)
            if len(nouns) >= 2:
                all_sentences.append(nouns)
        done += 1
        if done % 500 == 0:
            print(
                f"  Tokenized {done}/{len(unique_texts)} texts "
                f"({len(all_sentences)} sentences so far)",
                flush=True,
            )

    print(
        f"  Total: {len(all_sentences)} sentences from {len(unique_texts)} texts",
        flush=True,
    )
    return all_sentences


def train_word2vec(
    sentences: list[list[str]],
    save_dir: str = MODEL_DIR,
) -> Word2Vec:
    """Train a skip-gram Word2Vec model on tokenized legal sentences."""
    print(
        f"\nTraining Word2Vec (dim={EMBED_DIM}, window={WINDOW}, "
        f"epochs={EPOCHS})...",
        flush=True,
    )

    model = Word2Vec(
        sentences=sentences,
        vector_size=EMBED_DIM,
        window=WINDOW,
        min_count=MIN_COUNT,
        workers=4,
        epochs=EPOCHS,
        sg=1,  # skip-gram
    )

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    model_path = os.path.join(save_dir, "word2vec_legal.model")
    model.save(model_path)
    print(f"  Saved to {model_path}")
    print(f"  Vocabulary: {len(model.wv)} words")
    return model


def train_fasttext(
    sentences: list[list[str]],
    save_dir: str = MODEL_DIR,
) -> FastText:
    """Train a skip-gram FastText model on tokenized legal sentences."""
    print(
        f"\nTraining FastText (dim={EMBED_DIM}, window={WINDOW}, "
        f"epochs={EPOCHS})...",
        flush=True,
    )

    model = FastText(
        sentences=sentences,
        vector_size=EMBED_DIM,
        window=WINDOW,
        min_count=MIN_COUNT,
        workers=4,
        epochs=EPOCHS,
        sg=1,
        min_n=2,
        max_n=6,
    )

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    model_path = os.path.join(save_dir, "fasttext_legal.model")
    model.save(model_path)
    print(f"  Saved to {model_path}")
    print(f"  Vocabulary: {len(model.wv)} words")
    return model


def demo_embeddings(model: Word2Vec | FastText, name: str) -> None:
    """Print sample nearest neighbours to verify embedding quality."""
    test_words = ["채권", "소유권", "계약", "손해", "판결"]
    print(f"\n=== {name}: Sample nearest neighbors ===")
    for word in test_words:
        if word in model.wv:
            neighbors = model.wv.most_similar(word, topn=5)
            neighbor_str = ", ".join(
                f"{w}({s:.2f})" for w, s in neighbors
            )
            print(f"  {word} -> {neighbor_str}")
        else:
            print(f"  {word} -> (not in vocabulary)")


def main() -> None:
    """Entry point: collect text, train both embedding models, and demo."""
    print("Collecting legal text from dataset...", flush=True)
    sentences = collect_all_sentences()

    w2v = train_word2vec(sentences)
    ft = train_fasttext(sentences)

    demo_embeddings(w2v, "Word2Vec")
    demo_embeddings(ft, "FastText")

    print(f"\nDone! Models saved to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
