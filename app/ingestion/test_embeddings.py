from sentence_transformers import SentenceTransformer

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Text that we want to convert into an embedding
text = "Java is an object-oriented programming language."

# Generate embedding
embedding = model.encode(text)

print(embedding)
print("Number of dimensions:", len(embedding))