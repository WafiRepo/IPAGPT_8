import os
import re
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
    # Set allow_dangerous_deserialization=True if the source is trusted
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Retrieve similar documents using FAISS vector store
def get_similar_docs(question):
    db = load_faiss_index()
    return db.similarity_search(question)

# Check if the response contains mathematical expressions or numbers
def contains_math(response):
    # Regular expression to match numbers, equations, or mathematical symbols
    math_pattern = r'[=+\-*/^(){}\[\]\\]|[0-9]'
    return re.search(math_pattern, response)

# Automatically format response with LaTeX if mathematical content is found
def format_response(response):
    if contains_math(response):
        st.latex(response)  # Render mathematical content as LaTeX
        st.write("Description: This equation represents the relationship between the variables shown.")
    else:
        st.write(response)  # Render normal content as text

# Get conversational chain for Google Generative AI
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    If the answer is not in the provided context, use the generative model from Gemini Pro. 
    If the response contains a numerical result, formula, or equation, display it using LaTeX format 
    and include a detailed explanation of the symbols and their meaning.\n\n
    Context:\n {context}\n
    Question: \n{question}\n
    Answer:
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
    # Search for relevant documents in FAISS index
    docs = get_similar_docs(question)
    
    # If relevant documents are found, use the conversational chain
    if docs:
        chain = get_conversational_chain()
        response = chain.invoke({"input_documents": docs, "question": question})
        
        # If the response contains the default prompt for no answer, fallback to generative model
        if "answer is not available in the context" in response['output_text']:
            return generate_fallback_response(question)
        else:
            return response['output_text']
    else:
        # Fallback to using a generative model if no relevant document is found
        return generate_fallback_response(question)

# Fallback response if no relevant document is found
def generate_fallback_response(question):
    # Use the free "chat-bison-001" model instead of "gemini-pro"
    response = genai.generate_text(prompt=question, model="chat-bison-001")
    return response.result

# Main app logic
def main():
    st.set_page_config(page_title="Chat PDF + Open Knowledge", layout="wide")
    st.header("Chat with PDF + Answer Beyond the Docs")

    # Load and process PDF
    pdf_directory = os.path.join(os.path.dirname(__file__), "pdf_files")
    pdf_path = os.path.join(pdf_directory, "IPA-BS-KLS-VIII.pdf")
    
    pdf_text = load_and_process_pdf(pdf_path)

    # Display the PDF text if it's not empty or None
    if pdf_text:
        st.write("**Processed PDF Text (Preview):**")
        st.write(pdf_text[:500])  # Show the first 500 characters
    else:
        st.warning("The PDF file is empty or could not be processed.")

    # Get user input (prompt)
    user_question = st.text_input("Ask a Question")

    if user_question:
        with st.spinner("Processing your request..."):
            answer = process_question(user_question)
            st.success("Response generated!")
            format_response(answer)  # Automatically format response based on content type

if __name__ == "__main__":
    main()
