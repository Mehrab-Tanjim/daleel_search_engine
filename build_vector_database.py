import pickle
import numpy as np
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
import os

def save_vectorstore(documents, embeddings, base_path):

    if not documents:
        raise ValueError("No valid documents to index")

    # Create FAISS vector store
    db = FAISS.from_documents(documents, embeddings, normalize_L2=True, distance_strategy = "MAX_INNER_PRODUCT")
    db.save_local(base_path)
    print(f"Saved vector store to {base_path}")
    return db

def build_vd(embedding_model_name, device):
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={"device": device, "trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True, "max_seq_length": 8192}  # Reduced batch size
    )

    for doctype in ['original', 'preprocessed']:
        
        model_path = f"{embedding_model_name.split('/')[-1]}_{doctype}_{device}"
        os.makedirs(f"vector_databases/{model_path}/", exist_ok=True)

        # Load documents
        with open(f"datasets/{doctype}_quran_docs.pkl", "rb") as f:
            quran_docs = pickle.load(f)
        with open(f"datasets/{doctype}_hadith_docs.pkl", "rb") as f:
            hadith_docs = pickle.load(f)

        save_vectorstore(quran_docs, embeddings, f"vector_databases/{model_path}/quran")
        save_vectorstore(hadith_docs, embeddings, f"vector_databases/{model_path}/hadith")
        # save_vectorstore(quran_docs + hadith_docs, embeddings, f"vector_databases/{model_path}/all")

if __name__ == "__main__":
    
    # Build and save vector stores
    embedding_model_names = [
        "fine_tuned_models/islamqa_fine_tuned_all-mpnet-base-v2",
    #     "Alibaba-NLP/gte-multilingual-base", 
    # "sentence-transformers/all-mpnet-base-v2", 'sentence-transformers/LaBSE', 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2', "intfloat/multilingual-e5-base", 
    ]

    device = 'cpu' #'cuda'

    for embedding_model_name in embedding_model_names:
        build_vd(embedding_model_name, device)
    
    