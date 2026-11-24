from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import time
import math
import os
from numpy import dot
from numpy.linalg import norm
from typing import List
import json
from sentence_spans import canonical_span_config
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
    positions = {}
    for result in results:
        reference = _ayah_reference(result[0])
        if reference not in positions:
            positions[reference] = len(deduped)
            deduped.append(result)
        elif result[1] > deduped[positions[reference]][1]:
            deduped[positions[reference]] = result
    return deduped


def _boolean(value, name):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized not in ('false', 'true'):
        raise ValueError(f"{name} must be 'false' or 'true'")
    return normalized == 'true'


def _translation_id(value, name='EXTRA_TRANSLATION_ID'):
    normalized = str(value).strip().lower()
    if normalized not in ('none', 'pickthall_1930'):
        raise ValueError(f"{name} must be 'none' or 'pickthall_1930'")
    return normalized


def _ayah_sort_key(reference):
    surah, ayah = reference.split(':')
    return int(surah), int(ayah)


REPRESENTATION_SCOPES = {
    'translation': ('translation',),
    'ai': ('ai',),
    'translation_ai': ('translation', 'ai'),
    'en_tafsir': ('en_tafsir',),
    'translation_en_tafsir': ('translation', 'en_tafsir'),
    'ai_en_tafsir': ('ai', 'en_tafsir'),
    'translation_ai_en_tafsir': ('translation', 'ai', 'en_tafsir'),
}
EXTRA_TRANSLATION_SCOPES = {
    'extra_translation': ('extra_translation',),
    'translation_extra_translation': ('translation', 'extra_translation'),
    'ai_extra_translation': ('ai', 'extra_translation'),
    'en_tafsir_extra_translation': ('en_tafsir', 'extra_translation'),
    'translation_ai_extra_translation': ('translation', 'ai', 'extra_translation'),
    'translation_en_tafsir_extra_translation': ('translation', 'en_tafsir', 'extra_translation'),
    'ai_en_tafsir_extra_translation': ('ai', 'en_tafsir', 'extra_translation'),
    'translation_ai_en_tafsir_extra_translation': ('translation', 'ai', 'en_tafsir', 'extra_translation'),
}
REPRESENTATION_PRIORITY = ('translation', 'ai', 'en_tafsir', 'extra_translation')


def _fuse_by_ayah(candidates_by_field, k, field_priority=REPRESENTATION_PRIORITY):
    priority = {field: index for index, field in enumerate(field_priority)}
    best = {}
    for field, candidates in candidates_by_field.items():
        if field not in priority:
            raise ValueError(f'Unknown representation field: {field}')
        for doc, score in candidates:
            reference = _ayah_reference(doc)
            current = best.get(reference)
            if current is None or score > current[1] or (score == current[1] and priority[field] < current[2]):
                best[reference] = (doc, score, priority[field])
    ordered = sorted(
        best.items(),
        key=lambda item: (-item[1][1], item[1][2], _ayah_sort_key(item[0]))
    )
    return [(doc, score) for _, (doc, score, _) in ordered[:k]]


class VectorSearchDeployment:
    def __init__(self, index_path, model_name, device, ai_index_path=None, en_tafsir_index_path=None, extra_translation_index_path=None):
        #Load the data from faiss
        st = time.time()
        self.model_name = model_name
        self.retrieval_mode = os.environ.get('RETRIEVAL_MODE', 'similarity')
        self.fetch_k = _positive_int(os.environ.get('FETCH_K', '50'), 'FETCH_K')
        self.mmr_lambda = float(os.environ.get('MMR_LAMBDA', '1.0'))
        self.span_mode, self.span_sentences = canonical_span_config(
            os.environ.get('SPAN_MODE', 'whole'), os.environ.get('SPAN_SENTENCES', '1')
        )
        self.semantic_ai = _boolean(os.environ.get('SEMANTIC_AI', 'false'), 'SEMANTIC_AI')
        if self.semantic_ai and not ai_index_path:
            raise ValueError('SEMANTIC_AI=true requires ai_index_path')
        self.semantic_en_tafsir = _boolean(os.environ.get('SEMANTIC_EN_TAFSIR', 'false'), 'SEMANTIC_EN_TAFSIR')
        if self.semantic_en_tafsir and not en_tafsir_index_path:
            raise ValueError('SEMANTIC_EN_TAFSIR=true requires en_tafsir_index_path')
        self.extra_translation_id = _translation_id(os.environ.get('EXTRA_TRANSLATION_ID', 'none'))
        if self.extra_translation_id != 'none' and not extra_translation_index_path:
            raise ValueError('EXTRA_TRANSLATION_ID=pickthall_1930 requires extra_translation_index_path')
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
        self.ai_db = None
        self.ai_db_docstore_id_to_index = {}
        self.en_tafsir_db = None
        self.en_tafsir_db_docstore_id_to_index = {}
        self.extra_translation_db = None
        self.extra_translation_db_docstore_id_to_index = {}
        if ai_index_path:
            self.ai_db = FAISS.load_local(ai_index_path, self.embeddings, normalize_L2=True, allow_dangerous_deserialization=True)
            self.ai_db_docstore_id_to_index = dict(zip(self.ai_db.index_to_docstore_id.values(), self.ai_db.index_to_docstore_id.keys()))
        if en_tafsir_index_path:
            self.en_tafsir_db = FAISS.load_local(en_tafsir_index_path, self.embeddings, normalize_L2=True, allow_dangerous_deserialization=True)
            self.en_tafsir_db_docstore_id_to_index = dict(zip(self.en_tafsir_db.index_to_docstore_id.values(), self.en_tafsir_db.index_to_docstore_id.keys()))
        if extra_translation_index_path:
            self.extra_translation_db = FAISS.load_local(extra_translation_index_path, self.embeddings, normalize_L2=True, allow_dangerous_deserialization=True)
            self.extra_translation_db_docstore_id_to_index = dict(zip(self.extra_translation_db.index_to_docstore_id.values(), self.extra_translation_db.index_to_docstore_id.keys()))
        et = time.time() - st
        print(f'Loading database took {et} seconds.')

    def _representation_store(self, field):
        if field == 'translation':
            return self.db, self.db_docstore_id_to_index
        if field == 'ai':
            return self.ai_db, self.ai_db_docstore_id_to_index
        if field == 'en_tafsir':
            return self.en_tafsir_db, self.en_tafsir_db_docstore_id_to_index
        if field == 'extra_translation':
            return getattr(self, 'extra_translation_db', None), getattr(self, 'extra_translation_db_docstore_id_to_index', {})
        raise ValueError(f'Unknown representation field: {field}')

    def search(self, method, query, k=10, return_embeddings=False,
                retrieval_mode=None, fetch_k=None, mmr_lambda=None,
                semantic_ai=None, representation_scope=None, semantic_en_tafsir=None, extra_translation_id=None,
                span_mode=None, span_sentences=None):
        if method not in ('best_match', 'best_match_dedup', 'mmr'):
            raise ValueError(f"Unknown search method: {method}")
        active_span_mode, active_span_sentences = canonical_span_config(
            getattr(self, 'span_mode', 'whole') if span_mode is None else span_mode,
            getattr(self, 'span_sentences', 1) if span_sentences is None else span_sentences,
        )
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
            ai_enabled = self.semantic_ai if semantic_ai is None else _boolean(semantic_ai, 'semantic_ai')
            tafsir_enabled = (
                getattr(self, 'semantic_en_tafsir', False)
                if semantic_en_tafsir is None else _boolean(semantic_en_tafsir, 'semantic_en_tafsir')
            )
            extra_id = (
                getattr(self, 'extra_translation_id', 'none')
                if extra_translation_id is None else _translation_id(extra_translation_id, 'extra_translation_id')
            )
            default_fields = ['translation']
            if ai_enabled:
                default_fields.append('ai')
            if tafsir_enabled:
                default_fields.append('en_tafsir')
            if extra_id != 'none':
                default_fields.append('extra_translation')
            scope = representation_scope or '_'.join(default_fields)
            if scope in REPRESENTATION_SCOPES:
                fields = REPRESENTATION_SCOPES[scope]
            elif scope in EXTRA_TRANSLATION_SCOPES:
                fields = EXTRA_TRANSLATION_SCOPES[scope]
            else:
                raise ValueError('Unknown representation_scope')
            if 'ai' in fields and not ai_enabled:
                raise ValueError('AI representation scopes require semantic_ai=true')
            if 'en_tafsir' in fields and not tafsir_enabled:
                raise ValueError('Tafsir representation scopes require semantic_en_tafsir=true')
            if 'extra_translation' in fields and extra_id == 'none':
                raise ValueError('Extra translation scopes require extra_translation_id=pickthall_1930')
            for field in fields:
                if self._representation_store(field)[0] is None:
                    raise ValueError(f'{field} representation scope requires a {field} index')
            if fields != ('translation',) and mode != 'similarity':
                raise ValueError('Optional representations are only measured with similarity retrieval')

            query_embedding = self.embeddings.embed_query(query)
            if scope == 'translation':
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
            else:
                candidates_by_field = {}
                for field in fields:
                    database, _ = self._representation_store(field)
                    candidates_by_field[field] = database.similarity_search_with_score_by_vector(
                        query_embedding, k=candidate_depth
                    )
                results = _fuse_by_ayah(candidates_by_field, k)[:k]

        if active_span_mode == 'whole' and any(doc.metadata.get('span_kind') == 'sentence' for doc, _ in results):
            raise ValueError('SPAN_MODE=whole cannot return sentence spans')
        if any(
            doc.metadata.get('span_kind') == 'sentence'
            and doc.metadata.get('span_sentences') not in (None, active_span_sentences)
            for doc, _ in results
        ):
            raise ValueError('Retrieved sentence span size does not match SPAN_SENTENCES')
        
        if return_embeddings:
            # Get embeddings for the results from the index
            result_embeddings = []
            for doc, score in results:
                # Reconstruct from the representation that supplied the winning score.
                field = doc.metadata.get('field', 'translation')
                database, id_to_index = self._representation_store(field)
                doc_embedding = database.index.reconstruct(id_to_index[doc.id])
                result_embeddings.append(doc_embedding)
            return results, result_embeddings
        
        return results

    def display_document(self, doc):
        """Return the retained translation for display while preserving retrieval attribution on doc."""
        field = doc.metadata.get('field', 'translation')
        if field not in ('ai', 'en_tafsir', 'extra_translation') and doc.metadata.get('span_kind') != 'sentence':
            return doc
        reference = _ayah_reference(doc)
        if not hasattr(self, 'translation_docs_by_ayah'):
            self.translation_docs_by_ayah = {}
            for doc_id in self.db.index_to_docstore_id.values():
                candidate = self.db.docstore.search(doc_id)
                if candidate is not None and candidate.metadata.get('field', 'translation') == 'translation' and candidate.metadata.get('span_kind') in (None, 'whole'):
                    self.translation_docs_by_ayah[_ayah_reference(candidate)] = candidate
        display = self.translation_docs_by_ayah.get(reference)
        if display is None or display.metadata.get('field', 'translation') != 'translation':
            raise ValueError(f'Retained display translation is missing for ayah: {reference}')
        return display

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
