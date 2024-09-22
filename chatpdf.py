import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY is not set")

genai.configure(api_key=api_key)

# Variabel global untuk menyimpan riwayat percakapan
conversation_history = []

# Fungsi untuk membaca teks dari PDF di direktori
def get_pdf_text_from_directory(directory):
    text = ""
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "rb") as f:
                pdf_reader = PdfReader(f)
                for page in pdf_reader.pages:
                    text += page.extract_text()
    return text

# Fungsi untuk memecah teks menjadi chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Fungsi untuk membuat vector store dari chunks teks
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Fungsi untuk mendapatkan model percakapan
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. 
    If the answer is not in the provided context just say, 'answer is not available in the context'. Do not provide the wrong answer.
    \n\n
    Conversation History: {history}\n\n
    Context:\n {context}\n
    Question: \n{question}\n
    Answer:
    """
    
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["history", "context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

# Fungsi untuk memproses input pengguna
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Load FAISS index
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    # Gabungkan riwayat percakapan dengan pertanyaan pengguna
    conversation_text = "\n".join(conversation_history)  # Menggabungkan riwayat prompt
    
    # Dapatkan chain percakapan
    chain = get_conversational_chain()

    # Proses pertanyaan dengan riwayat
    response = chain({"input_documents": docs, "history": conversation_text, "question": user_question}, return_only_outputs=True)

    # Simpan pertanyaan dan jawaban dalam riwayat
    conversation_history.append(f"User: {user_question}")
    conversation_history.append(f"Assistant: {response['output_text']}")

    st.write("Reply: ", response["output_text"])

# Fungsi utama
def main():
    st.set_page_config(page_title="Chat PDF", layout="wide")
    st.header("Chat with PDF using Gemini💁")

    # Input pertanyaan dari pengguna
    user_question = st.text_input("Ask a Question from the PDF Files")

    # Proses file PDF yang sudah ada di folder
    directory = os.path.join(os.path.dirname(__file__), "pdf_files")
    raw_text = get_pdf_text_from_directory(directory)

    # Proses PDF menjadi chunks
    text_chunks = get_text_chunks(raw_text)
    
    # Membuat dan menyimpan vector store
    get_vector_store(text_chunks)

    if user_question:
        user_input(user_question)

    st.sidebar.title("Conversation History:")
    st.sidebar.write("\n".join(conversation_history))  # Menampilkan riwayat percakapan di sidebar

if __name__ == "__main__":
    main()
