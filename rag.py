from langchain_cohere import CohereRerank
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import  RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_classic.retrievers import EmbedchainRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers import BM25Retriever


embedding  =  HuggingFaceEmbeddings(
    model_name = "BAAI/bge-m3"
)

def document_loader(path):

    loader = PyPDFLoader(path)
    try:
        docs =  loader.load()
        print(f"Length of page : {docs}")
        splitter =  RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap =  100
        )

        chunks = splitter.split_documents(docs)
        print(f"Length of chunk : {chunks}")
    except Exception as e:
        raise ValueError("Error while loading documents")
    
    return chunks



path  =  input("Enter the path of policy pdf : ")
docs =document_loader(path)








