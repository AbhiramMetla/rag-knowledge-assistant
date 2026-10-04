from ingestion.document_loader import load_text_file
from ingestion.text_splitter import split_text


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


if __name__ == "__main__":
    main()