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
            st.error(f"The file {pdf_path} is empty.")
            return None

        with open(pdf_path, "rb") as f:
            pdf_reader = PdfReader(f)
            for i, page in enumerate(pdf_reader.pages):
                if i >= limit:
                    break
                text += page.extract_text()

    except FileNotFoundError:
        st.error(f"File not found: {pdf_path}")
        return None
    except PdfReadError:
        st.error(f"Cannot read the PDF file. It may be corrupted: {pdf_path}")
        return None
    except Exception as e:
        st.error(f"An error occurred: {e}")
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

# Automatically format all responses with LaTeX if necessary
def format_response(response):
    try:
        # Assume LaTeX formatting is required and attempt to render with st.latex()
        st.latex(response)  # Render mathematical content as LaTeX
        st.write("Description: This equation represents the relationship between the variables shown.")
    except Exception:
        # If LaTeX rendering fails, fallback to plain text
        st.write(response)

# Get conversational chain for Google Generative AI (with Chain of Thought)
def get_conversational_chain():
    prompt_template = """
    Answer the question by first breaking down your reasoning step by step (using the Chain of Thought approach). 
    Explain your thought process before arriving at the final answer. \n\n
    Context:\n {context}\n
    Question: \n{question}\n
    Step-by-step reasoning and answer:
    """
    
    # Create the LLM Chain
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    # Initialize the LLMChain with model and prompt
    llm_chain = LLMChain(llm=model, prompt=prompt)

    # Specify the document_variable_name explicitly
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
    st.set_page_config(page_title="Chat PDF + Open Knowledge", layout="wide")
    st.header("Chat with PDF + Answer Beyond the Docs")

    pdf_directory = os.path.join(os.path.dirname(__file__), "pdf_files")
    pdf_path = os.path.join(pdf_directory, "IPA-BS-KLS-VIII.pdf")
    
    pdf_text = load_and_process_pdf(pdf_path)

    if pdf_text:
        st.write("**Processed PDF Text (Preview):**")
        st.write(pdf_text[:500])
    else:
        st.warning("The PDF file is empty or could not be processed.")

    user_question = st.text_input("Ask a Question")

    if user_question:
        with st.spinner("Processing your request..."):
            answer = process_question(user_question)
            st.success("Response generated!")
            format_response(answer)  # Automatically attempt LaTeX formatting for all responses

if __name__ == "__main__":
    main()
