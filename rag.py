import os
from langchain_classic.retrievers import ensemble
from langchain_cohere import CohereRerank
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import  RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers import BM25Retriever


embedding  =  HuggingFaceEmbeddings(
    model_name = "BAAI/bge-m3"
)

def document_loader(path):

    loader = PyPDFLoader(path)
    try:
        docs =  loader.load()
        print(f"Length of page : {len(docs)}")
        splitter =  RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap =  100
        )

        chunks = splitter.split_documents(docs)
        print(f"Length of chunk : {len(chunks)}")
    except Exception as e:
        raise ValueError("Error while loading documents")
    
    return chunks



path  =  input("Enter the path of policy pdf : ")
docs =document_loader(path)

def hybrid_search(docs : str):

    try:
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embedding,
            persist_directory="./chroma_pdr",
            collection_name="hybrid_demo",
        )

    except Exception as e:
        raise ValueError("Error while creating vectorstore")

    
    vector_search = vectorstore.as_retriever(
        search_type =  "mmr",
        search_kwargs = {
            "k" : 5,
            "fetch_k" : 10,
            "lambda_mult:" : 0.6
        }
    )

    bm25_search = BM25Retriever.from_documents(
        documents=docs,
        k = 5
    )

    Ensembleretriever=  EnsembleRetriever(
        weights=[0.5, 0.5],
        retrievers=[vector_search, bm25_search]
    )

    Reranker =  CohereRerank(
        model="rerank-v4.0-pro",
        top_n=6,
        cohere_api_key=os.getenv("COHERE_API_KEY")
    )


    retriever = ContextualCompressionRetriever(
        base_compressor=Reranker,
        base_retriever=Ensembleretriever
    )

    return retriever



    









