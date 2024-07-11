from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISSIndex
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
import os


load_dotenv() # take environment variables from .env.

# Create  LLM model
llm = GoogleGenerativeAI(model="gemini-pro",google_api_key=os.environ["GOOGLE_API_KEY"], temperature=0.1)

# # Initialize instructor embeddings using the Hugging Face model
instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
vectordb_file_path = "faiss_index"


def create_vector_db():
    # Load data 
    loader = CSVLoader(file_path='D:\GENERATIVE-AI\ED-TECH\DATA\data_faqs.csv', source_column="prompt")
    data = loader.load()

    # Create a text splitter for chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )

    # Split the documents into chunks
    chunks = text_splitter.split_documents(data)

# Create a vector store using FAISS
    vectordb = FAISS.from_documents(documents=chunks,
                                    embedding=instructor_embeddings)
   

# Save vector database locally
    vectordb.save_local(vectordb_file_path)




