import os
from pathlib import Path
import numpy as np
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

EMBEDDING_MODEL = "text-embedding-3-small"
TOP_K = 2
MIN_SIMILARITY = 0.30
MIN_WORDS = 8  # drop bare titles/headers (not real content to retrieve)

# A snippet must also score at least this fraction of the best match to be
# kept. A single-topic question scores its runner-up around 0.5x the top
# match (loosely related, not worth passing on), while a question that
# genuinely spans two sections scores its second around 0.8x.
RELATIVE_CUTOFF = 0.65

# Resolved from this file's own location (not the current working directory)
# so the tool works the same whether you run `python main.py` from the repo
# root, from inside src/, or from a PyCharm run configuration.
DEFAULT_KB_PATH = str(Path(__file__).resolve().parent.parent / "data" / "knowledge_base.txt")

_embedder = None
_paragraph_cache: dict = {}

def get_embedder() -> OpenAIEmbeddings:
    global _embedder
    if _embedder is None:
        _embedder = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return _embedder


def load_paragraphs(file_path: str) -> list[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    return [p for p in paragraphs if len(p.split()) >= MIN_WORDS]


def get_paragraph_embeddings(file_path: str) -> tuple[list[str], np.ndarray]:
    """Loads and embeds the knowledge base once, then reuses the result for
    every subsequent call in this process (keeps embedding cost minimal)."""
    if file_path not in _paragraph_cache:
        paragraphs = load_paragraphs(file_path)
        vectors = np.array(get_embedder().embed_documents(paragraphs), dtype=np.float32)
        _paragraph_cache[file_path] = (paragraphs, vectors)
    return _paragraph_cache[file_path]


@tool
def search_knowledge_base(query: str, file_path: str = DEFAULT_KB_PATH) -> str:
    """Reads the knowledge base file and returns the paragraphs most
    semantically similar to the query, using OpenAI embeddings + cosine
    similarity. Returns up to the top 2 matching snippets, or a "no match"
    message if nothing is similar enough."""
    if not os.path.exists(file_path):
        return f"Error: {file_path} not found."

    paragraphs, doc_vectors = get_paragraph_embeddings(file_path)
    query_vector = np.array(get_embedder().embed_query(query), dtype=np.float32)

    # sklearn compares two 2D matrices, so wrap the single query vector in a
    # 1-row matrix and take that row back out of the result.
    similarities = cosine_similarity(query_vector.reshape(1, -1), doc_vectors)[0]
    ranked_indices = similarities.argsort()[::-1]

    # Each snippet is tagged with its cosine-similarity score so the ranking
    # is visible in the output (and to the Report Generator).
    best_score = similarities[ranked_indices[0]]
    top_matches = [
        f"[similarity {similarities[i]:.3f}]\n{paragraphs[i]}"
        for i in ranked_indices[:TOP_K]
        if similarities[i] >= MIN_SIMILARITY
        and similarities[i] >= best_score * RELATIVE_CUTOFF
    ]

    if not top_matches:
        return "No directly matching snippets found in the knowledge base."

    return "\n---\n".join(top_matches)

