"""Load html from files, clean up, split, ingest into Weaviate."""
import pickle

from langchain.document_loaders.csv_loader import  CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
import os
os.environ["OPENAI_API_KEY"] = "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"

def ingest_docs():
    """Get documents from web pages."""
    # loader = ReadTheDocsLoader("langchain.readthedocs.io/en/latest/")
    loader = CSVLoader("/home/swaingotnochill/hackathon/being-genai-hackathon/convertcsv.csv")
    raw_documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    documents = text_splitter.split_documents(raw_documents)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Save vectorstore
    with open("vectorstore.pkl", "wb") as f:
        pickle.dump(vectorstore, f)


if __name__ == "__main__":
    ingest_docs()
