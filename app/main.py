from ingestion.document_loader import load_text_file


def main():
    file_path = "data/sample.txt"

    content = load_text_file(file_path)

    print("Document loaded successfully!")
    print()
    print(content)


if __name__ == "__main__":
    main()