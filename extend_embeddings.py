from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import os
from time import sleep

# Configuration
file_name = "mentalhealth_db"
data_path = "data/new_info"  # Directory of new documents to be added
docs = []
new_doc_paths = os.listdir(f"{data_path}")

# Load new documents
print("Loading new documents...")
for file in new_doc_paths:
    print("Loading file: ", file)
    if file.endswith(".pdf"):
        pdf_path = f"./{data_path}/" + file
        loader = PyPDFLoader(pdf_path)
        docs.extend(loader.load())
        print("PDF loaded: ", file, "length: ", len(loader.load()))
    elif file.endswith('.docx'):
        doc_path = f"./{data_path}/" + file
        loader = Docx2txtLoader(doc_path)
        print("Docx loaded: ", file, "length: ", len(loader.load()))
        docs.extend(loader.load())
        
print("Total number of new docs loaded: ", len(docs))

# Split text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400, add_start_index=True)
splits = text_splitter.split_documents(docs)

batch_size = 1000  # Adjust batch size if needed
batched_docs = [splits[i:i + batch_size] for i in range(0, len(splits), batch_size)]

# Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=os.environ["OPENAI_KEY"])

# Initialize existing Chroma collection
chroma_db = Chroma(
    persist_directory="data/chroma_db/docs_db_openai_2000_chunk",
    collection_name=file_name,
    embedding_function=embeddings
)

# Check if the collection is not empty
existing_collection = chroma_db.get()  # Retrieve existing collection data

if len(existing_collection['ids']) != 0:
    # Add new documents to the existing collection
    print("Adding new documents to the collection...")
    for batch in batched_docs:
        chroma_db.add_documents(batch)
        print("Batch inserted...")
        sleep(60)  # To respect rate limits if necessary

    # Persist the changes to disk
    chroma_db.persist()
    print("New documents added and database updated.")