from ingestion.document_loader import load_text_file
from ingestion.text_splitter import split_text
from sentence_transformers import SentenceTransformer
import chromadb

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

    # Create a ChromaDB client
    client = chromadb.PersistentClient(path="./chroma_db")

    # Create or get our collection
    collection = client.get_or_create_collection(
        name="rag_knowledge"
    )

    # Store the chunks and their embeddings
    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings.tolist()
    )

    print("\nChunks stored in ChromaDB successfully!")
    print("Number of documents in collection:", collection.count())

    question = "Where are the embeddings stored?"

    # Search ChromaDB
    results = collection.query(
        query_embeddings=[embedding_model.encode(question).tolist()],
        n_results=2
    )

    print("\n--- Retrieved Chunks ---")

    for document in results["documents"][0]:
        print(document)

if __name__ == "__main__":
    main()