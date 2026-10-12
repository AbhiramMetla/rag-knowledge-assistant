
import streamlit as st
from rag_pipeline import ask_question

# Configure the web page
st.set_page_config(
    page_title="RAG Knowledge Assistant",
    page_icon="📚",
    layout="centered"
)

st.title("📚 RAG Knowledge Assistant")
st.caption("Ask questions about your documents.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept a new question
if question := st.chat_input("Ask a question about your document..."):

    # Display and save the user's question
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    # Retrieve context and generate an answer
    with st.chat_message("assistant"):
        with st.spinner("Searching your document and generating an answer..."):
            try:
                answer = ask_question(question)
            except Exception as error:
                answer = (
                    f"Something went wrong: {error}\n\n"
                    "Check your terminal for details."
                )

        st.markdown(answer)

    # Save the assistant's response
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
