import streamlit as st
from PyPDF2 import PdfReader
import os
import concurrent.futures
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Load environment variables from .env
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY is not set")

# Configure Google Generative AI
import google.generativeai as genai
genai.configure(api_key=api_key)

# Cache PDF loading and processing to avoid reloading each time
@st.cache_data
def load_and_process_pdf(pdf_path, limit=5):
    text = ""
    with open(pdf_path, "rb") as f:
        pdf_reader = PdfReader(f)
        for i, page in enumerate(pdf_reader.pages):
            if i >= limit:  # Only process the first 'limit' pages for performance
                break
            text += page.extract_text()
    return text

# Cache FAISS index loading
@st.cache_resource
def load_faiss_index():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # Set allow_dangerous_deserialization to True, as you trust the source
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)


# Retrieve similar documents using FAISS vector store
def get_similar_docs(question):
    db = load_faiss_index()
    return db.similarity_search(question)

# Get conversational chain for Google Generative AI
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    If the answer is not in the provided context, just say, 'answer is not available in the context'.\n\n
    Context:\n {context}\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Process prompt and fetch response, with parallel processing
def process_prompt_with_parallel(prompts):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(process_single_prompt, prompts))
    return results

# Function to handle single prompt
def process_single_prompt(prompt):
    similar_docs = get_similar_docs(prompt)
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": similar_docs, "question": prompt},
        return_only_outputs=True
    )
    return response['output_text']

# Manage session state to cache conversation history
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Add question and answer to session state
def add_to_conversation(question, answer):
    st.session_state.conversation_history.append({"question": question, "answer": answer})

# Display conversation history in the app
def display_conversation():
    if st.session_state.conversation_history:
        for idx, entry in enumerate(st.session_state.conversation_history):
            st.write(f"**User {idx+1}:** {entry['question']}")
            st.write(f"**Assistant {idx+1}:** {entry['answer']}")

# Main app logic
def main():
    st.set_page_config(page_title="Optimized Chat PDF", layout="wide")
    st.header("Chat with PDFs using Google Generative AI")

    # Load and process PDF once, and cache results
    pdf_directory = os.path.join(os.path.dirname(__file__), "pdf_files")
    pdf_text = load_and_process_pdf(os.path.join(pdf_directory, "IPA-BS-KLS-VIII.pdf"))

    # Show part of the processed PDF text
    st.write("**Processed PDF Text (Preview):**")
    st.write(pdf_text[:500])  # Show the first 500 characters

    # Get user input (prompt)
    user_question = st.text_input("Ask a Question based on the PDF")

    # Display conversation history
    display_conversation()

    # Handle user input and processing
    if user_question:
        with st.spinner("Processing your request..."):
            answer = process_single_prompt(user_question)
            add_to_conversation(user_question, answer)

        st.success("Response generated!")
        st.write(f"**Assistant:** {answer}")

if __name__ == "__main__":
    main()
