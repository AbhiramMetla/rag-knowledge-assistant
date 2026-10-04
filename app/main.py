from ingestion.document_loader import load_text_file
from ingestion.text_splitter import split_text
from sentence_transformers import SentenceTransformer


def main():
    file_path = "data/sample.txt"

    content = load_text_file(file_path)

    print("Document loaded successfully!")
    print(f"Total characters: {len(content)}")

    chunks = split_text(content)

    print(f"Total chunks: {len(chunks)}")
    print()

    for index, chunk in enumerate(chunks, start=1):
        print(f"--- Chunk {index} ---")
        print(chunk)
        print()

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Generate embeddings for all chunks
    embeddings = embedding_model.encode(chunks)

    print("\nEmbeddings generated successfully!")
    print("Number of embeddings:", len(embeddings))
    print("Dimensions of each embedding:", len(embeddings[0]))


if __name__ == "__main__":
    main()