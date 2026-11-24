from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import time
import math
import os
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


def _positive_int(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if isinstance(value, float) and value != parsed:
        raise ValueError(f"{name} must be a positive integer")
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def _ayah_reference(doc):
    try:
        surah = int(doc.metadata['surah'])
        ayah = int(doc.metadata['ayah'])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('Retrieved document is missing a valid ayah ID') from exc
    if surah <= 0 or ayah <= 0:
        raise ValueError('Retrieved document is missing a valid ayah ID')
    return f'{surah}:{ayah}'


def _deduplicate_by_ayah(results):
    deduped = []
    seen = set()
    for result in results:
        reference = _ayah_reference(result[0])
        if reference not in seen:
            seen.add(reference)
            deduped.append(result)
    return deduped


class VectorSearchDeployment:
    def __init__(self, index_path, model_name, device):
        #Load the data from faiss
        st = time.time()
        self.model_name = model_name
        self.retrieval_mode = os.environ.get('RETRIEVAL_MODE', 'similarity')
        self.fetch_k = _positive_int(os.environ.get('FETCH_K', '50'), 'FETCH_K')
        self.mmr_lambda = float(os.environ.get('MMR_LAMBDA', '1.0'))
        if self.retrieval_mode not in ('similarity', 'mmr'):
            raise ValueError("RETRIEVAL_MODE must be 'similarity' or 'mmr'")
        if not math.isfinite(self.mmr_lambda) or not 0 <= self.mmr_lambda <= 1:
            raise ValueError('MMR_LAMBDA must be finite and between 0 and 1')
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

    def search(self, method, query, k=10, return_embeddings=False,
                retrieval_mode=None, fetch_k=None, mmr_lambda=None):
        if method not in ('best_match', 'best_match_dedup', 'mmr'):
            raise ValueError(f"Unknown search method: {method}")
        if method == 'best_match_dedup':
            if retrieval_mode is not None:
                raise ValueError('best_match_dedup does not accept retrieval_mode')
            results = self.relevance_search_dedup(query, k)
        else:
            k = _positive_int(k, 'k')
            mode = retrieval_mode or ('mmr' if method == 'mmr' else self.retrieval_mode)
            candidate_depth = _positive_int(
                self.fetch_k if fetch_k is None else fetch_k, 'fetch_k'
            )
            relevance_weight = float(
                self.mmr_lambda if mmr_lambda is None else mmr_lambda
            )
            if mode not in ('similarity', 'mmr'):
                raise ValueError("retrieval_mode must be 'similarity' or 'mmr'")
            if candidate_depth < k:
                raise ValueError('fetch_k must be at least k')
            if not math.isfinite(relevance_weight) or not 0 <= relevance_weight <= 1:
                raise ValueError('mmr_lambda must be finite and between 0 and 1')

            query_embedding = self.embeddings.embed_query(query)
            if mode == 'similarity':
                candidates = self.db.similarity_search_with_score_by_vector(
                    query_embedding,
                    k=candidate_depth
                )
            else:
                candidates = self.db.max_marginal_relevance_search_with_score_by_vector(
                    query_embedding,
                    k=k,
                    fetch_k=candidate_depth,
                    lambda_mult=relevance_weight
                )
            results = _deduplicate_by_ayah(candidates)[:k]
        
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
        """Return unique ayah IDs; the legacy embedding threshold is not applied."""
        del similarity_threshold
        k = _positive_int(k, 'k')
        return self.search(
            method='best_match',
            query=query,
            k=k,
            retrieval_mode='similarity',
            fetch_k=max(k, self.fetch_k)
        )[:k]


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
