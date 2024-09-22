import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from dotenv import load_dotenv
from PyPDF2.errors import PdfReadError

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY is not set")

# Configure Google Generative AI
import google.generativeai as genai
genai.configure(api_key=api_key)

# Load PDF and process it into text
@st.cache_data
def load_and_process_pdf(pdf_path, limit=5):
    text = ""
    try:
        # Check if the file is empty before processing
        if os.path.getsize(pdf_path) == 0:
            st.error(f"Berkas {pdf_path} kosong.")
            return None

        with open(pdf_path, "rb") as f:
            pdf_reader = PdfReader(f)
            for i, page in enumerate(pdf_reader.pages):
                if i >= limit:
                    break
                text += page.extract_text()

    except FileNotFoundError:
        st.error(f"Berkas tidak ditemukan: {pdf_path}")
        return None
    except PdfReadError:
        st.error(f"Tidak dapat membaca berkas PDF. Mungkin berkas rusak: {pdf_path}")
        return None
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
        return None

    return text

@st.cache_resource
def load_faiss_index():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Retrieve similar documents using FAISS vector store
def get_similar_docs(question):
    db = load_faiss_index()
    return db.similarity_search(question)

# Format response without using LaTeX
def format_response(response):
    st.write(response)

# Get conversational chain for Google Generative AI (with Chain of Thought)
def get_conversational_chain():
    prompt_template = """
    Jawab pertanyaan dengan menjelaskan proses berpikir Anda langkah demi langkah (menggunakan pendekatan Chain of Thought). 
    Jelaskan proses berpikir Anda sebelum sampai pada jawaban akhir. 
    Jika pertanyaan melibatkan hasil numerik, rumus, atau persamaan, jelaskan setiap langkah dan berikan penjelasan terperinci 
    tentang setiap simbol dan langkah dalam persamaan.\n\n
    Konteks:\n {context}\n
    Pertanyaan: \n{question}\n
    Langkah-langkah dan jawaban:
    """
    
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    llm_chain = LLMChain(llm=model, prompt=prompt)
    chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="context")
    
    return chain

# Generate response based on FAISS or fallback to generative model
def process_question(question):
    docs = get_similar_docs(question)
    
    if docs:
        chain = get_conversational_chain()
        response = chain.invoke({"input_documents": docs, "question": question})
        
        if "answer is not available in the context" in response['output_text']:
            return generate_fallback_response(question)
        else:
            return response['output_text']
    else:
        return generate_fallback_response(question)

# Fallback response if no relevant document is found
def generate_fallback_response(question):
    response = genai.generate_text(prompt=question, model="gemini-pro")
    return response.result

# Main app logic
def main():
    st.set_page_config(page_title="Ilmu Pengetahuan Alam (IPA) Kelas VIII - SMPN 1 Buay Madang Timur", layout="wide")
    st.header("Ilmu Pengetahuan Alam (IPA) Kelas VIII - SMPN 1 Buay Madang Timur")

    pdf_directory = os.path.join(os.path.dirname(__file__), "pdf_files")
    pdf_path = os.path.join(pdf_directory, "IPA-BS-KLS-VIII.pdf")
    
    pdf_text = load_and_process_pdf(pdf_path)

    if pdf_text:
        # Static title of the book
        st.write(f"**Judul Buku: Ilmu Pengetahuan Alam**")
        st.write(f"Diterbitkan Oleh Pusat Perbukuan\nBadan Standar, Kurikulum, dan Asesmen Pendidikan\n"
                 f"Kementerian Pendidikan, Kebudayaan, Riset, dan Teknologi\n"
                 f"Komplek Kemdikbudristek Jalan RS. Fatmawati, Cipete, Jakarta Selatan\n"
                 f"https://buku.kemdikbud.go.id\nDisusun Oleh Okky Fajar Tri Maryana, dkk")
    else:
        st.warning("Berkas PDF kosong atau tidak dapat diproses.")

    user_question = st.text_input("Ajukan Pertanyaan")

    if user_question:
        with st.spinner("Sedang memproses permintaan Anda..."):
            answer = process_question(user_question)
            st.success("Respons berhasil dihasilkan!")
            format_response(answer)

if __name__ == "__main__":
    main()
