import streamlit as st
from langchain_ollama import OllamaLLM
import chromadb
from chromadb.utils import embedding_functions

# Initialize global variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_files_content" not in st.session_state:
    st.session_state.uploaded_files_content = []

# Initialize ChromaDB Client and Collection
def initialize_chromadb():
    try:
        client = chromadb.Client()
        ef = embedding_functions.OpenAIEmbeddingFunction(api_key="YOUR_API_KEY")  # Replace with your OpenAI API key
        collection = client.get_or_create_collection(name="chatbot_memory", embedding_function=ef)
        return collection
    except Exception as e:
        st.error(f"Error initializing ChromaDB: {e}")
        return None

# Save conversation history to ChromaDB
def save_to_chromadb(collection, user_input, response):
    try:
        collection.add(
            documents=[user_input, response],
            metadatas=[{"role": "user"}, {"role": "bot"}],
            ids=[f"user-{len(st.session_state.chat_history)}", f"bot-{len(st.session_state.chat_history)}"]
        )
    except Exception as e:
        st.error(f"Error saving to ChromaDB: {e}")

# Retrieve past context from ChromaDB
def retrieve_from_chromadb(collection, query):
    try:
        results = collection.query(query_texts=[query], n_results=1)
        if results and results.get('documents'):
            return results['documents'][0]  # Return the most relevant document
        return None
    except Exception as e:
        st.error(f"Error retrieving context from ChromaDB: {e}")
        return None

# Function to generate a response
def generate_response(model_name, prompt):
    try:
        llm = OllamaLLM(model=model_name)
        response = llm.invoke(prompt)
        return response
    except Exception as e:
        return f"Error: {e}"

# Main function for Streamlit app
def main():
    st.title("Ollama Chatbot with ChromaDB Memory")

    # Initialize ChromaDB
    collection = initialize_chromadb()
    if collection is None:
        st.warning("ChromaDB could not be initialized. Contextual memory is disabled.")

    # Sidebar for model selection
    st.sidebar.header("Settings")
    model_choice = st.sidebar.selectbox(
        "Choose a model:",
        ("llama3.2:1b", "llama3.1:8b")
    )

    # Clear chat history button
    if st.sidebar.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.uploaded_files_content = []
        if collection:
            try:
                collection.delete(where={})  # Clear ChromaDB collection
            except Exception as e:
                st.error(f"Error clearing ChromaDB: {e}")

    # File uploader for additional context (multiple files)
    uploaded_files = st.file_uploader("Upload .txt files for context:", type="txt", accept_multiple_files=True)
    if uploaded_files:
        st.session_state.uploaded_files_content = []
        for file in uploaded_files:
            try:
                content = file.read().decode("utf-8")
                st.session_state.uploaded_files_content.append(content)
            except Exception as e:
                st.error(f"Error processing file {file.name}: {e}")

        st.success(f"Successfully uploaded {len(uploaded_files)} file(s).")

    # Display uploaded file contents for user visibility
    if st.session_state.uploaded_files_content:
        st.write("### Uploaded Files Content")
        for i, content in enumerate(st.session_state.uploaded_files_content):
            st.text_area(f"File {i+1} Content:", value=content, height=150, key=f"file_context_display_{i}")

    # Input box for user prompt
    user_input = st.text_input("Enter your message:", key="user_input")

    if user_input:
        # Combine all file contents and user input for context
        file_context = "\n\n".join(st.session_state.uploaded_files_content) if st.session_state.uploaded_files_content else ""
        combined_context = f"File Context:\n{file_context}\nUser: {user_input}" if file_context else user_input

        # Retrieve context from ChromaDB
        chromadb_context = retrieve_from_chromadb(collection, user_input) if collection else None
        if chromadb_context and file_context:
            prompt = f"ChromaDB Context: {chromadb_context}\n{combined_context}"
        elif chromadb_context:
            prompt = f"ChromaDB Context: {chromadb_context}\nUser: {user_input}"
        else:
            prompt = combined_context

        # Debugging prompt for verification
        st.write("### Final Prompt Sent to Model")
        st.text_area("Prompt:", value=prompt, height=150)

        # Generate and append response
        response = generate_response(model_choice, prompt)
        st.session_state.chat_history.append(f"You: {user_input}")
        st.session_state.chat_history.append(f"Bot: {response}")

        # Save to ChromaDB
        if collection:
            save_to_chromadb(collection, user_input, response)

    # Display chat history
    st.write("### Chat History")
    if st.session_state.chat_history:
        st.text_area(
            "Conversation:",
            value="\n".join(st.session_state.chat_history),
            height=300,
            key="chat_history_display"
        )

# Run the app
if __name__ == "__main__":
    main()
