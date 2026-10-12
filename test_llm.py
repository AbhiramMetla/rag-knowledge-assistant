
from langchain_ollama import ChatOllama

# Initialize the locally running language model
llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)

# Send a test question
response = llm.invoke("Explain Retrieval-Augmented Generation in two sentences.")

# Print the model's answer
print(response.content)
