from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import time
import math
from numpy import dot
from numpy.linalg import norm
from typing import List

class VectorSearchDeployment:
    def __init__(self, index_path, model_name, device):
        #Load the data from faiss
        st = time.time()
        
        model_kwargs = {'device': device, 'trust_remote_code':True}
        encode_kwargs = {'normalize_embeddings': True, "max_seq_length": 8192}
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

        self.db = FAISS.load_local(index_path, self.embeddings, normalize_L2=True, allow_dangerous_deserialization=True)
        et = time.time() - st
        print(f'Loading database took {et} seconds.')

    def search(self, method, query, k=10): 
        if method == 'best_match':
            results = self.db.similarity_search_with_score_by_vector(
                self.embeddings.embed_query(query),
                k=k
            )
        elif method == 'best_match_dedup':
            results = self.relevance_search_dedup(query, k)
        elif method == 'mmr':
            results = self.db.max_marginal_relevance_search_with_score_by_vector(self.embeddings.embed_query(query), k=k, fetch_k=k*2)
        
        return results

    def relevance_search_dedup(self, query, k=10, similarity_threshold=0.95):
        """
        Search for most relevant results with deduplication
        Args:
            query (str): Search query
            k (int): Number of results to return
            similarity_threshold (float): Cosine similarity threshold for deduplication (0-1)
        Returns:
            list: Deduplicated search results with scores
        """
        # Get initial results using similarity search (more than k to allow for dedup)
        initial_k = k * 3
        results = self.db.similarity_search_with_score_by_vector(
            self.embeddings.embed_query(query),
            k=initial_k
        )
        
        # Deduplication process
        deduped_results = []
        seen_embeddings = []
        
        for doc, score in results:
            doc_embedding = self.embeddings.embed_query(doc.page_content)
            is_duplicate = False
            
            # Check similarity with already selected documents
            for seen_emb in seen_embeddings:
                # Calculate cosine similarity
                cos_sim = dot(doc_embedding, seen_emb) / (norm(doc_embedding) * norm(seen_emb))
                if cos_sim > similarity_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                deduped_results.append((doc, score))
                seen_embeddings.append(doc_embedding)
            
            # Stop when we have enough unique results
            if len(deduped_results) >= k:
                break
        
        return deduped_results[:k]

def normalize_l2(score):
    return 1 - score/math.sqrt(2)


def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))


def semantic_match_score(reference: str, candidates: List[str], embed):
    ref_emb = embed(reference)
    scores = []
    for candidate in candidates:
        cand_emb = embed(candidate)
        scores.append(cosine_similarity(ref_emb, cand_emb))
    return max(scores) if scores else 0.0
