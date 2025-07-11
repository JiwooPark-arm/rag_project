import os
import re
from pathlib import Path
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import tiktoken
import logging

# Configure logging for better visibility and control
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()

# Retrieve API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

# Initialize embedding model
embeddings = OpenAIEmbeddings(openai_api_key=api_key)

# Define the path to your codebase
codebase_path = Path("verilator")  # Update this path as needed

# Initialize the tokenizer
# 'cl100k_base' is suitable for models like text-embedding-ada-002
encoder = tiktoken.get_encoding("cl100k_base")

def preprocess_code(code_str: str, extension: str) -> str:
    """
    Preprocesses code by removing comments and excessive whitespace based on file extension.
    """
    # Remove comments based on file extension
    if extension in [".ts", ".js", ".c", ".cpp", ".h"]:
        # Remove multi-line comments
        code_str = re.sub(r"/\*[\s\S]*?\*/", "", code_str)
        # Remove single-line comments
        code_str = re.sub(r"//.*", "", code_str)
    elif extension == ".py":
        # Remove single-line comments
        code_str = re.sub(r"#.*", "", code_str)
        # Remove multi-line docstrings
        code_str = re.sub(r'"""[\s\S]*?"""', "", code_str)
        code_str = re.sub(r"'''[\s\S]*?'''", "", code_str)
    elif extension == ".sh":
        # Remove single-line comments
        code_str = re.sub(r"#.*", "", code_str)

    # Remove excessive whitespace
    code_str = re.sub(r"\s+", " ", code_str)
    return code_str

def load_code_files(path: Path, extensions: list = [".txt", ".yml", ".cfg", ".py", ".c", ".cpp", ".h", ".vc", ".svh", ".sv"]) -> list:
    """
    Loads and preprocesses code files from the specified path.
    Supports recursive directory traversal.
    """
    documents = []
    if path.is_dir():
        # Recursively find all files with the given extensions
        for ext in extensions:
            for file in path.rglob(f"*{ext}"):
                logging.info(f"Processing file: {file}")
                try:
                    with open(file, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                        # Preprocess the code
                        content = preprocess_code(content, ext)
                        # Append as a Document
                        documents.append(Document(page_content=content, metadata={"source": str(file)}))
                except Exception as e:
                    logging.error(f"Error reading {file}: {e}")
    elif path.is_file() and path.suffix in extensions:
        logging.info(f"Processing file: {path}")
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                # Preprocess the code
                content = preprocess_code(content, path.suffix)
                # Append as a Document
                documents.append(Document(page_content=content, metadata={"source": str(path)}))
        except Exception as e:
            logging.error(f"Error reading {path}: {e}")
    else:
        logging.warning(f"Path {path} is neither a directory nor a supported file.")
    return documents

def count_tokens(chunks: list, encoder) -> int:
    """
    Counts the total number of tokens in all chunks using the specified encoder.
    """
    total_tokens = 0
    for chunk in chunks:
        tokens = encoder.encode(chunk.page_content)
        total_tokens += len(tokens)
    return total_tokens

def main():
    # Step 1: Load code documents
    logging.info("Loading code files...")
    documents = load_code_files(codebase_path)
    logging.info(f"Loaded {len(documents)} documents.")

    # Step 2: Split into chunks
    logging.info("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,       # Adjust based on your needs
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    logging.info(f"Created {len(chunks)} chunks.")

    # Step 3: Count tokens in all chunks
    logging.info("Counting tokens in chunks...")
    total_tokens = count_tokens(chunks, encoder)
    logging.info(f"Total tokens consumed: {total_tokens}")

    # Step 4: Create vector store
    logging.info("Creating vector store...")
    vector_store = FAISS.from_documents(chunks, embeddings)
    logging.info("Vector store created.")

    # Step 5: Save the vector store to disk
    index_path = "faiss_index.pkl"
    vector_store.save_local(index_path)
    logging.info(f"Vector store saved to {index_path}.")

if __name__ == "__main__":
    main()