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
    Jawab pertanyaan dengan nada yang ramah dan mudah dipahami, seperti berbicara dengan seorang teman. 
    Jika pertanyaan melibatkan matematika atau analisis numerik, berikan penjelasan langkah demi langkah 
    yang terstruktur dengan cara yang santai namun tetap jelas.
    Gunakan contoh nyata jika memungkinkan, dan pastikan setiap langkah dijelaskan dengan baik.
    Untuk pertanyaan lainnya, jawab secara ringkas namun lengkap.\n\n
    Konteks:\n {context}\n
    Pertanyaan: \n{question}\n
    Jawaban Terstruktur:
    """
    
    # Create the LLM Chain
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5)  # Moderate temperature for creativity and clarity
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
            return generate_fallback_response(question)  # Switch to generative AI
        else:
            return response['output_text']
    else:
        return generate_fallback_response(question)  # If no docs found, fallback

# Fallback response if no relevant document is found
def generate_fallback_response(question):
    # Use Google Generative AI to answer questions beyond the documents
    try:
        response = genai.generate_text(prompt=question, model="gemini-pro")
        return response.result
    except Exception as e:
        return f"Terjadi kesalahan saat memanggil Google Generative AI: {e}"

# Main app logic
def main():
    st.set_page_config(page_title="Ilmu Pengetahuan Alam (IPA) Kelas VIII - SMPN 1 Buay Madang Timur", layout="wide")

    # Define the path for a single logo using os.path.join
    logo_path_1 = os.path.join(os.path.dirname(__file__), "images/SMPN1BMT logo.jpeg")

    # CSS for making images responsive and centered
    st.markdown("""
        <style>
        .center-logo {
            display: flex;
            justify-content: center;
        }
        .center-logo img {
            width: 100%;
            max-width: 150px; /* Adjust the max width of the logo here */
            height: auto;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Display a single logo in the center
    st.image(logo_path_1, use_column_width=False)

    st.header("Ilmu Pengetahuan Alam (IPA) Kelas VIII - SMPN 1 Buay Madang Timur")

    pdf_directory = os.path.join(os.path.dirname(__file__), "pdf_files")
    pdf_path = os.path.join(pdf_directory, "IPA-BS-KLS-VIII.pdf")
    
    pdf_text = load_and_process_pdf(pdf_path)

    if pdf_text:
        st.write(f"**Nama Buku: Ilmu Pengetahuan Alam**")
        st.write(f"Diterbitkan Oleh Pusat Perbukuan\nBadan Standar, Kurikulum, dan Asesmen Pendidikan\n"
                 f"Kementerian Pendidikan, Kebudayaan, Riset, dan Teknologi\n"
                 f"Komplek Kemdikbudristek Jalan RS. Fatmawati, Cipete, Jakarta Selatan\n"
                 f"https://buku.kemdikbud.go.id")
    else:
        st.warning("Berkas PDF kosong atau tidak dapat diproses.")

    # Single input text box for questions
    user_question = st.text_input("Ajukan Pertanyaan", placeholder="Ketik pertanyaan Anda di sini...", label_visibility="visible")

    if user_question:
        with st.spinner("Sedang memproses permintaan Anda..."):
            answer = process_question(user_question)
            st.success("Respons berhasil dihasilkan!")
            format_response(answer)


if __name__ == "__main__":
    main()
