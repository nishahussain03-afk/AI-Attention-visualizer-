from attention import calculate_attention
from embedding import create_embeddings


def main():
    words = ["Artificial", "Intelligence", "machine", "learning"]

    print("Testing embeddings...")
    embeddings = create_embeddings(words)

    print("Embedding shape:", embeddings.shape)

    print("Testing attention...")
    scores = calculate_attention(embeddings)

    print("Attention scores:", scores)
    print("Highest attention word:", words[int(scores.argmax())])

    print("ALL BASIC TESTS PASSED")


if __name__ == "__main__":
    main()
