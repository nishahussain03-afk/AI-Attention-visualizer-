import hashlib
import numpy as np


EMBEDDING_DIMENSION = 384


def word_to_embedding(word):
    """
    Convert a word into a deterministic 384-dimensional
    numerical vector using NumPy.

    This is a lightweight educational embedding method.
    It does not require PyTorch or Sentence Transformers.
    """

    word = word.lower().strip()

    if not word:
        return np.zeros(EMBEDDING_DIMENSION, dtype=np.float64)

    seed_bytes = hashlib.sha256(
        word.encode("utf-8")
    ).digest()

    seed = int.from_bytes(
        seed_bytes[:8],
        byteorder="little",
        signed=False
    )

    rng = np.random.default_rng(seed)

    vector = rng.standard_normal(
        EMBEDDING_DIMENSION
    )

    norm = np.linalg.norm(vector)

    if norm != 0:
        vector = vector / norm

    return vector.astype(np.float64)


def create_embeddings(words):
    """
    Create one 384-dimensional numerical embedding
    for every word.
    """

    if not words:
        return np.empty(
            (0, EMBEDDING_DIMENSION),
            dtype=np.float64
        )

    embeddings = np.array(
        [word_to_embedding(word) for word in words],
        dtype=np.float64
    )

    return embeddings