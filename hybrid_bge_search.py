import streamlit as st
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
import numpy as np
import torch
import faiss
import pickle
import json

# Load model and indices
model = SentenceTransformer("BAAI/bge-m3")  # or locally saved version
faiss_index = faiss.read_index("index/faiss_index.faiss")
with open("index/id2doc.pkl", "rb") as f:
    id2doc = pickle.load(f)
with open("index/token_embeddings.json", "r") as f:
    token_embeddings = json.load(f)

es = Elasticsearch("http://localhost:9200")  # ensure ES is running locally
index_name = "bge-m3-sparse"

def dense_search(query, top_k=10):
    dense_vec = model.encode(query, normalize_embeddings=True)
    D, I = faiss_index.search(np.array([dense_vec]), top_k)
    return [(id2doc[i], float(D[0][idx])) for idx, i in enumerate(I[0])]

def sparse_search(query, top_k=10):
    sparse = model.encode(query, output_value="sparse_embedding")
    indices = sparse["indices"]
    values = sparse["values"]
    tokens = [str(i) for i in indices]

    script_query = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": """
                    double score = 0.0;
                    for (token in params.sparse_weights.keySet()) {
                        if (doc[token].size() != 0) {
                            score += doc[token].value * params.sparse_weights[token];
                        }
                    }
                    return score;
                """,
                "params": {"sparse_weights": {str(k): float(v) for k, v in zip(indices, values)}}
            }
        }
    }

    resp = es.search(index=index_name, body={"size": top_k, "query": script_query})
    return [(hit["_source"]["text"], hit["_score"]) for hit in resp["hits"]["hits"]]

def colbert_score(q_reps, p_reps):
    q = torch.tensor(q_reps)
    p = torch.tensor(p_reps)
    token_scores = torch.einsum("in,jn->ij", q, p)
    scores, _ = token_scores.max(-1)
    return torch.sum(scores) / q.size(0)

def colbert_rerank(query, candidates):
    q_reps = model.encode(query, output_value="colbert")
    reranked = []
    for doc in candidates:
        pid = str(hash(doc[0]))  # or use actual ID mapping
        if pid in token_embeddings:
            p_reps = np.array(token_embeddings[pid])
            score = colbert_score(q_reps, p_reps).item()
            reranked.append((doc[0], score))
    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked

st.title("🔍 Hybrid Search with BGE-M3")
query = st.text_input("Enter your query:")

if query:
    st.subheader("Dense Search Results")
    dense_results = dense_search(query)
    for text, score in dense_results:
        st.write(f"**Score**: {score:.4f}")
        st.write(text)
        st.markdown("---")

    st.subheader("Sparse Search Results (Elasticsearch)")
    sparse_results = sparse_search(query)
    for text, score in sparse_results:
        st.write(f"**Score**: {score:.4f}")
        st.write(text)
        st.markdown("---")

    st.subheader("ColBERT Reranked Results")
    merged = {doc: score for doc, score in dense_results + sparse_results}
    colbert_results = colbert_rerank(query, list(merged.items()))
    for text, score in colbert_results[:10]:
        st.write(f"**ColBERT Score**: {score:.4f}")
        st.write(text)
        st.markdown("---")
