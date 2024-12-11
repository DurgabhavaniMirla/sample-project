from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface  import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

if __name__ == "__main__":

    print("Loading Documents...")
    
    loader = TextLoader("speech.txt")
    documents = loader.load()
    print(f"Loaded {len(documents)} documents")

    print("Splitting Documents...")
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    split_documents = splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(split_documents)} chunks")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("Inserting Documents into VectorDB...")
    vector_db = PineconeVectorStore.from_documents(split_documents, embeddings, index_name="pinecone")
    print(f"Inserted {len(split_documents)} documents into vectorDB")