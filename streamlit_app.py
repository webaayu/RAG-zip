import os
import streamlit as st
import fitz  # PyMuPDF
import zipfile
import io
from langchain_community.llms import Ollama
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup

# Ensure pysqlite3 is imported and used
import pysqlite3
import pysqlite3.dbapi2 as sqlite3
os.environ["SQLITE_LIBRARY_PATH"] = pysqlite3.__file__

# Function to get response from LLM
def get_llm_response(input, content, prompt):
    try:
        model = Ollama(model='llama2')
        cont = str(content)
        response = model.invoke([input, cont, prompt])
        return response
    except Exception as e:
        st.error(f"Error occurred while connecting to LLM: {e}")
        return ""

# Function to extract text from PDF file
def extract_text_from_pdf(file):
    try:
        with fitz.open(stream=file, filetype="pdf") as doc:
            text = ""
            for page in doc:
                text += page.get_text()
            return text
    except Exception as e:
        st.error(f"Error occurred while processing PDF: {e}")
        return ""

# Function to extract text from HTML file
def extract_text_from_html(file):
    try:
        soup = BeautifulSoup(file, 'html.parser')
        return soup.get_text()
    except Exception as e:
        st.error(f"Error occurred while processing HTML: {e}")
        return ""

# Function to extract text from text file
def extract_text_from_txt(file):
    try:
        return file.read().decode("utf-8")
    except Exception as e:
        st.error(f"Error occurred while processing text file: {e}")
        return ""

# Main function
def main():
    st.title("ZIP File Chatbot")

    st.sidebar.title("Upload ZIP File")
    uploaded_file = st.sidebar.file_uploader("Choose a ZIP file", type=['zip'])

    prompt = st.text_input("Ask a Question", "")

    submitted = st.button("Submit")

    if submitted:
        if uploaded_file is not None:
            bytes_data = uploaded_file.read()
            zip_file = io.BytesIO(bytes_data)

            extracted_texts = []
            with zipfile.ZipFile(zip_file, 'r') as z:
                for file_info in z.infolist():
                    with z.open(file_info) as file:
                        if file_info.filename.endswith('.pdf'):
                            pdf_text = extract_text_from_pdf(file.read())
                            if pdf_text:
                                extracted_texts.append(pdf_text)
                        elif file_info.filename.endswith('.html') or file_info.filename.endswith('.htm'):
                            html_text = extract_text_from_html(file.read())
                            if html_text:
                                extracted_texts.append(html_text)
                        elif file_info.filename.endswith('.txt'):
                            txt_text = extract_text_from_txt(file.read())
                            if txt_text:
                                extracted_texts.append(txt_text)

            combined_text = "\n".join(extracted_texts)
            
            if combined_text:
                try:
                    embeddings = HuggingFaceEmbeddings()

                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=20,
                        length_function=len
                    )
                    chunks = text_splitter.split_text(combined_text)

                    persist_directory = 'file_embeddings'
                    vectordb = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=persist_directory)
                    vectordb.persist()
                    st.write("Embeddings stored successfully in ChromaDB.")
                    st.write(f"Persist directory: {persist_directory}")

                    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
                    st.write(vectordb)

                    if prompt:
                        docs = vectordb.similarity_search(prompt)
                        st.write(docs[0])
                        text = docs[0]
                        input_prompt = """You are an expert in understanding text contents. You will receive input files and you will have to answer questions based on the input files."""
                        response = get_llm_response(input_prompt, text, prompt)
                        st.subheader("Generated Answer:")
                        st.write(response)
                except Exception as e:
                    st.error(f"Error occurred during text processing: {e}")

if __name__ == "__main__":
    main()
