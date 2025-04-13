from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import time
import math
from numpy import dot
from numpy.linalg import norm
from typing import List
import json
class LLMResponseIndex:
    def __init__(self, llm_output_path, name):
        with open(llm_output_path, 'r', encoding='utf-8') as f:
            llm_output = json.load(f)

        self.index = {}
        self.name = name
        for d in llm_output:
            self.index[d['question'].strip()] = [dic for dic in d['openai_response_for_references'] if dic['type'].lower()==self.name.lower()]

    def search(self, method, query, k=10):
        refs = self.index.get(query.strip(), [])
        if not refs:
            return []
        
        results = []
        for ref in refs:
            text = ref['text']
            if self.name.lower() == 'quran':
                reference = ref['reference'].split()[-1].split(':')
                metadata = {'SurahNo': reference[0], 'AyahNo': reference[1]}
                
            elif self.name.lower() == 'hadith':
                reference = ref['reference'].split()
                metadata = {'source': ref['reference'].split()[0], 
                            'chapter_no': '', 
                            'hadith_no': ref['reference'].split()[-1]}
            
            # format like FAISS: (doc, score)
            # Fake score of 1.0 for all, metadata is minimal   
            results.append((SimpleDoc(text, metadata), 1.0))
        
        return results[:k]  # Return top k results
        
class SimpleDoc:
    def __init__(self, text, metadata):
        self.page_content = text
        self.metadata = metadata
        self.id = None  # Not needed unless you're matching IDs


class VectorSearchDeployment:
    def __init__(self, index_path, model_name, device):
        #Load the data from faiss
        st = time.time()
        self.model_name = model_name
        model_kwargs = {'device': device, 'trust_remote_code':True}
        encode_kwargs = {'normalize_embeddings': True, "max_seq_length": 8192}
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

        self.db = FAISS.load_local(index_path, self.embeddings, normalize_L2=True, allow_dangerous_deserialization=True)
        self.db_docstore_id_to_index = dict(zip(self.db.index_to_docstore_id.values(), self.db.index_to_docstore_id.keys()))
        et = time.time() - st
        print(f'Loading database took {et} seconds.')

    def search(self, method, query, k=10, return_embeddings=False): 
        query_embedding = self.embeddings.embed_query(query)
        if method == 'best_match':
            results = self.db.similarity_search_with_score_by_vector(
                query_embedding,
                k=k
            )
        elif method == 'best_match_dedup':
            results = self.relevance_search_dedup(query, k)
        elif method == 'mmr':
            results = self.db.max_marginal_relevance_search_with_score_by_vector(query_embedding, k=k, fetch_k=k*2)
        
        if return_embeddings:
            # Get embeddings for the results from the index
            result_embeddings = []
            for doc, score in results:
                # Get the document's embedding from the index
                doc_embedding = self.db.index.reconstruct(self.db_docstore_id_to_index[doc.id])
                result_embeddings.append(doc_embedding)
            return results, result_embeddings
        
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
        # Get initial results with embeddings using similarity search (more than k to allow for dedup)
        initial_k = k * 3
        results, embeddings = self.search(
            method='best_match',
            query=query,
            k=initial_k,
            return_embeddings=True
        )
        
        # Deduplication process
        deduped_results = []
        seen_embeddings = []
        
        for (doc, score), doc_embedding in zip(results, embeddings):
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
