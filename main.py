import os
import logging
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

load_dotenv()  # take environment variables from .env.

# Create LLM model
llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=os.environ["GOOGLE_API_KEY"], temperature=0.1)

# Initialize instructor embeddings using the Hugging Face model
instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
vectordb_file_path = "faiss_index"

def create_vector_db():
    # Load data 
    loader = CSVLoader(file_path=r'D:/GENERATIVE-AI/ED-TECH/DATA/data_faqs.csv', source_column="prompt")
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
    vectordb = FAISS.from_documents(documents=chunks, embedding=instructor_embeddings)

    # Save vector database locally
    vectordb.save_local(vectordb_file_path)
    print(f"Vector database created and saved to {vectordb_file_path}")

def get_qa_chain():
    # Check if the vector database file exists
    if not os.path.exists(os.path.join(vectordb_file_path, "index.faiss")):
        raise FileNotFoundError(f"Vector database file not found at {vectordb_file_path}")

    # Load the vector database from the local folder
    vectordb = FAISS.load_local(vectordb_file_path, instructor_embeddings, allow_dangerous_deserialization=True)

    # Create a retriever for querying the vector database
    retriever = vectordb.as_retriever(score_threshold=0.7)

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt": PROMPT})

    return chain

if __name__ == "__main__":
    create_vector_db()
    chain = get_qa_chain()
    print(chain("Do you have javascript course?"))
