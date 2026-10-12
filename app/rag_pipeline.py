
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_ollama import ChatOllama


# 1. Connect to our existing ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_collection(name="rag_knowledge")


# 2. Load the same embedding model used previously
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# 3. Initialize our local LLM
llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)


# 4. Retrieve relevant document chunks


def retrieve_context(question, top_k=2, max_distance=1.7):

    question_embedding = embedding_model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "distances"]
    )

    documents = results["documents"][0]
    distances = results["distances"][0]

    relevant_documents = []

    print("\n--- Retrieval Results ---")

    for i, (document, distance) in enumerate(
        zip(documents, distances), start=1
    ):
        print(f"\nChunk {i}")
        print(f"Distance: {distance:.4f}")
        print(f"Content: {document}")

        if distance <= max_distance:
            relevant_documents.append(document)
            print("Status: Accepted")
        else:
            print("Status: Rejected (distance too high)")

    print("-------------------------\n")

    return "\n\n".join(relevant_documents)




# 5. Generate an answer using the retrieved context
def generate_answer(question, context):

    prompt = f"""
You are a helpful document question-answering assistant.

Answer the user's question using only the provided context.
If the answer is not present in the context, clearly say:
"I could not find the answer in the provided document."

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    return response.content


# 6. Connect retrieval and generation
def ask_question(question):

    # Step 1: Retrieve relevant chunks from ChromaDB
    context = retrieve_context(question)

    if not context:
        return "No relevant information was found in the document."

    # Step 2: Display the retrieved context for debugging
    print("\n--- Retrieved Context ---")
    print(context)
    print("-------------------------\n")

    # Step 3: Generate an answer using the retrieved context
    answer = generate_answer(question, context)

    return answer


# 7. Run interactive question-answering
if __name__ == "__main__":

    print("RAG Knowledge Assistant")
    print("Type 'exit' to stop.\n")

    while True:
        question = input("Ask a question: ").strip()

        if question.lower() == "exit":
            print("Goodbye!")
            break

        if not question:
            continue

        answer = ask_question(question)

        print("\nAnswer:", answer, "\n")
