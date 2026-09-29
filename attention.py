import numpy as np


def softmax(x):
    """
    Numerically stable softmax.
    """
    x = np.asarray(x, dtype=np.float64)

    shifted = x - np.max(
        x,
        axis=-1,
        keepdims=True
    )

    exp_x = np.exp(shifted)

    return exp_x / np.sum(
        exp_x,
        axis=-1,
        keepdims=True
    )


def calculate_attention(X):
    """
    Calculate a classroom demonstration of scaled dot-product attention.

    X  = input embeddings
    Q  = Query representations
    K  = Key representations
    V  = Value representations

    The displayed word score is the average attention received
    by each word.
    """
    X = np.asarray(X, dtype=np.float64)

    if X.ndim != 2:
        raise ValueError("Embeddings must be a 2-dimensional array.")

    if X.shape[0] == 0:
        raise ValueError("At least one word is required.")

    d_model = X.shape[1]
    d_k = 64

    rng = np.random.default_rng(42)

    WQ = rng.standard_normal((d_model, d_k))
    WK = rng.standard_normal((d_model, d_k))
    WV = rng.standard_normal((d_model, d_k))

    Q = X @ WQ
    K = X @ WK
    V = X @ WV

    scores = Q @ K.T

    scaled_scores = scores / np.sqrt(d_k)

    attention_weights = softmax(scaled_scores)

    word_scores = attention_weights.mean(axis=0)

    return word_scores
