import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY is not set")

# Configure Google Generative AI
import google.generativeai as genai
genai.configure(api_key=api_key)

# Load PDF and Process it into text
@st.cache_data
def load_and_process_pdf(pdf_path, limit=5):
    text = ""
    with open(pdf_path, "rb") as f:
        pdf_reader = PdfReader(f)
        for i, page in enumerate(pdf_reader.pages):
            if i >= limit:  # Limit to first 'limit' pages
                break
            text += page.extract_text()
    return text

# Cache FAISS index loading
@st.cache_resource
def load_faiss_index():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return FAISS.load_local("faiss_index", embeddings)

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

# Generate response based on FAISS or fallback to generative model
def process_question(question):
    # Search for relevant documents in FAISS index
    docs = get_similar_docs(question)
    
    # If relevant documents are found, use the conversational chain
    if docs:
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
        return response['output_text']
    else:
        # Fallback to using a generative model if no relevant document is found
        return generate_fallback_response(question)

# Fallback response if no relevant document is found
def generate_fallback_response(question):
    # Using Google Generative AI directly to answer questions beyond the documents
    response = genai.generate_text(prompt=question, model="gemini-pro")
    return response.result

# Main app logic
def main():
    st.set_page_config(page_title="Chat PDF + Open Knowledge", layout="wide")
    st.header("Chat with PDF + Answer Beyond the Docs")

    # Load and process PDF once
    pdf_directory = os.path.join(os.path.dirname(__file__), "pdf_files")
    pdf_text = load_and_process_pdf(os.path.join(pdf_directory, "IPA-BS-KLS-VIII.pdf"))

    # Display part of the processed PDF text
    st.write("**Processed PDF Text (Preview):**")
    st.write(pdf_text[:500])  # Show the first 500 characters

    # Get user input (prompt)
    user_question = st.text_input("Ask a Question")

    if user_question:
        with st.spinner("Processing your request..."):
            answer = process_question(user_question)
            st.success("Response generated!")
            st.write(f"**Answer:** {answer}")

if __name__ == "__main__":
    main()
