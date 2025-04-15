import json
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch, helpers
from tqdm import tqdm

# Your corpus (replace this with your actual documents)
corpus = [
    "Allah is the most merciful.",
    "The Prophet Muhammad (PBUH) emphasized kindness to neighbors.",
    "Fasting during Ramadan is one of the five pillars of Islam.",
    # Add more documents here...
]

# Init model
model = SentenceTransformer("BAAI/bge-m3")

# Encode dense vectors
dense_embs = model.encode(corpus, normalize_embeddings=True)

# FAISS index
dim = dense_embs.shape[1]
faiss_index = faiss.IndexFlatIP(dim)
faiss_index.add(np.array(dense_embs))
faiss.write_index(faiss_index, "index/faiss_index.faiss")

# Save id2doc mapping
id2doc = {i: doc for i, doc in enumerate(corpus)}
with open("index/id2doc.pkl", "wb") as f:
    pickle.dump(id2doc, f)

# Generate and save ColBERT-style token embeddings
colbert_embs = model.encode(corpus, output_value="colbert")
token_embeddings = {str(hash(doc)): vec.tolist() for doc, vec in zip(corpus, colbert_embs)}
with open("index/token_embeddings.json", "w") as f:
    json.dump(token_embeddings, f)

# Create sparse vectors and upload to Elasticsearch
sparse_embs = model.encode(corpus, output_value="sparse_embedding")
es = Elasticsearch("http://localhost:9200")
index_name = "bge-m3-sparse"

# Create the ES index with dynamic mapping
if es.indices.exists(index=index_name):
    es.indices.delete(index=index_name)

mapping = {
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            # token_id fields will be added dynamically
        }
    }
}
es.indices.create(index=index_name, body=mapping)

# Index documents
bulk_data = []
for i, doc in tqdm(enumerate(corpus)):
    doc_id = str(i)
    sparse = sparse_embs[i]
    token_dict = {str(k): v for k, v in zip(sparse['indices'], sparse['values'])}
    es_doc = {"_index": index_name, "_id": doc_id, "_source": {"text": doc, **token_dict}}
    bulk_data.append(es_doc)

helpers.bulk(es, bulk_data)
print("✅ Indexing complete")
