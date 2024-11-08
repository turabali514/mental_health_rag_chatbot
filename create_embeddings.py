from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
import os
from langchain_community.document_loaders import Docx2txtLoader
from langchain_huggingface import HuggingFaceEmbeddings
from time import sleep


file_name = "mentalhealth_db"
data_path = "data/books"
docs = []
doc_paths = os.listdir(f"{data_path}")

# Create a List of Documnets from all of our files in the ./docs folder
print("Total number of docs paths: ", len(doc_paths))
for file in doc_paths:
    # if not file.endswith("Healing ADD Show 10 29 2013.docx"):
    #     continue
    print("Loading file: ", file)
    if file.endswith(".pdf"):
        pdf_path = f"./{data_path}/" + file
        loader = PyPDFLoader(pdf_path)
        docs.extend(loader.load())
        print("PDF loaded: ", file, "length: ", len(loader.load()))
    elif file.endswith('.docx'):
        doc_path = f"./{data_path}/" + file
        loader = Docx2txtLoader(doc_path)
        docs.extend(loader.load())
        print("Docx loaded: ", file, "length: ", len(loader.load()))
        
print("Total number of docs loaded: ", len(docs))

# Split text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400, add_start_index=True)
splits = text_splitter.split_documents(docs)

batch_size = 1000  # Adjust based on rate limits and token constraints
batched_docs = [splits[i:i + batch_size] for i in range(0, len(splits), batch_size)]

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=os.environ["OPENAI_KEY"])
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

chroma_db = Chroma(persist_directory="data/chroma_db", collection_name=file_name, embedding_function=embeddings)
# Get the collection from the Chroma database
collection = chroma_db.get()

# If the collection is empty, create a new one
if len(collection['ids']) == 0:
    print("Creating new embeddings...")
    for batch in batched_docs:
        chroma_db = Chroma.from_documents(
            documents=batch, 
            embedding=embeddings, 
            persist_directory="data/chroma_db",
            collection_name=file_name
        )
        print("Batch inserted...")
        sleep(60)

    # Save the Chroma database to disk
    chroma_db.persist()