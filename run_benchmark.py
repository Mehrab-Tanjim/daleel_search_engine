import json
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from utils import *  # Assuming VectorSearchDeployment is defined here
import re
from tqdm import tqdm
import logging
from datetime import datetime
import numpy as np
from multiprocessing import Pool
from functools import partial
import multiprocessing
from build_vector_database import build_vd
from utils import VectorSearchDeployment

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Compile regex once
COLON_PATTERN = re.compile(r'\b\d+:\d+\b')

def load_benchmark_data(path: str):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Failed to load benchmark data: {e}")
        raise

def extract_colon_reference(text):
    match = COLON_PATTERN.search(text)
    return match.group(0) if match else ''

def batch_semantic_match_score(ref_text, result_embeddings, embed_func):
    """Compute semantic similarity scores in batch"""
    ref_embedding = embed_func(ref_text)
    
    # Cosine similarity
    dot_product = np.dot(result_embeddings, ref_embedding)
    norms = np.linalg.norm(ref_embedding) * np.linalg.norm(result_embeddings, axis=1)
    similarities = dot_product / (norms + 1e-8)  # Avoid division by zero
    return np.max(similarities)

def evaluate_retrieval(results, ground_truth_refs, result_embeddings=None, matching='exact', eval_embed_func=None, sim_threshold=0.75):
    matched = 0
    retrieved = len(results) if results else 0
    total_refs = len(ground_truth_refs) if ground_truth_refs else 0

    if not results or not ground_truth_refs:
        return 0, total_refs, retrieved, 0.0, 0.0

    if matching == 'exact':
        results_lower = set([r.lower() for r in results])
        for ref in ground_truth_refs:
            if ref.lower() in results_lower:
                matched += 1
    else:
        # Need to perform fuzzy matching or semantic similarity for hadiths since the reference by book and hadith is not consistent
        # For example, "Muslim 8" vs "Muslim book 8" vs "Muslim 8:1234" vs "Muslim 1234" vs "Muslim 8/1234"
        for ref in ground_truth_refs:
            sim_text = batch_semantic_match_score(ref[0], result_embeddings, eval_embed_func)
            # sim_reference = batch_semantic_match_score(ref[1], [res[1] for res in results], embed_func)
            if sim_text >= sim_threshold: # or sim_reference >= sim_threshold
                matched += 1

    precision = matched / retrieved if retrieved > 0 else 0.0
    recall = matched / total_refs if total_refs > 0 else 0.0
    return matched, total_refs, retrieved, precision, recall

def save_results(results, model_name, doctype, device, output_dir="evaluation_results"):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/eval_{model_name.split('/')[-1]}_{doctype}_{device}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    logging.info(f"Results saved to {filename}")

def load_precomputed_embeddings(eval_index_path, eval_model_name, device):
    """Load precomputed evaluation embeddings from a FAISS index"""
    try:
        logging.info(f"Loading precomputed evaluation embeddings from {eval_index_path}")
        vd = VectorSearchDeployment(
                            eval_index_path, eval_model_name, device
                        )
        return vd
    except Exception as e:
        try:
            logging.warning(f"Failed to load precomputed evaluation embeddings from {eval_index_path}, building new index.")
            build_vd(eval_model_name, device)
            vd = VectorSearchDeployment(
                            eval_index_path, eval_model_name, device
                        )
            return vd
        except Exception as e:
            logging.error(f"Failed to load precomputed evaluation embeddings: {e}")
            raise

def process_entry(entry, name, model_faiss_index, method, k, sim_threshold, eval_embed_func, eval_faiss_index):
    query = entry["question"].strip()

    try:
        if name == 'quran':
            references = [extract_colon_reference(ref["reference"]) 
                          for ref in entry["extracted_references"] 
                          if ref['type'].lower() == name]
            
            if not references:
                return None
                
            retrieved_docs = model_faiss_index.search(method, query, k)
            retrieved_texts = [f"{doc.metadata['SurahNo']}:{doc.metadata['AyahNo']}" 
                              for doc, score in retrieved_docs]
        
            matched, expected, retrieved, precision, recall = evaluate_retrieval(
                retrieved_texts, references)

        else:
            references = [(ref["text"], ref["reference"]) 
                          for ref in entry["extracted_references"] 
                          if ref['type'].lower() == name]
            if not references:
                return None

            retrieved_docs = model_faiss_index.search(method, query, k)
            retrieved_texts = [(doc.page_content, 
                               ' '.join([doc.metadata['source'], 
                                        doc.metadata['chapter_no'], 
                                        doc.metadata['hadith_no']])) 
                              for doc, score in retrieved_docs]

            # Retrieve precomputed evaluation embeddings using document IDs
            retrieved_ids = [model_faiss_index.db_docstore_id_to_index[doc.id] for i, doc in enumerate([doc for doc, _ in retrieved_docs])]  # Use index as fallback ID
            retrieved_eval_embeddings = []
            for doc_id in retrieved_ids:
                try:
                    # Assuming eval_faiss_index stores embeddings in the same serial as model_faiss_index
                    # and that the document IDs are consistent across both indices
                    doc_vector = eval_faiss_index.db.index.reconstruct(int(doc_id))  # Fetch embedding by ID
                    retrieved_eval_embeddings.append(doc_vector)
                except Exception as e:
                    logging.warning(f"Could not fetch embedding for ID {doc_id}: {e}")
                    retrieved_eval_embeddings.append(np.zeros(eval_faiss_index.index.d))  # Fallback zero vector
            
            retrieved_eval_embeddings = np.array(retrieved_eval_embeddings)

            matched, expected, retrieved, precision, recall = evaluate_retrieval(
                retrieved_texts, references, retrieved_eval_embeddings, matching='cosine', 
                eval_embed_func=eval_embed_func, sim_threshold=sim_threshold)

        return {
            "query": query,
            "precision": precision,
            "recall": recall,
            "matched": matched,
            "ground_truths": references,
            "retrieved_text": retrieved_texts
        }

    except Exception as e:
        logging.error(f"Search or evaluation failed for query '{query}': {e}")
        return None

def run_benchmark(model_name, doctype, device, benchmark_path, method, k=5, sim_threshold=0.8, num_rows=None, eval_model_name="Alibaba-NLP/gte-multilingual-base" ):
    try:
        benchmark_data = load_benchmark_data(benchmark_path)
    except Exception as e:
        logging.error(f"Failed to load benchmark data: {e}")
        return


    base_path = f"vector_databases/{model_name.split('/')[-1]}_{doctype}_{device}"
    eval_base_path = f"vector_databases/{eval_model_name.split('/')[-1]}_{doctype}_{device}"
    index_paths = {
        name: os.path.join(base_path, name.lower())
        for name in [ "hadith", "quran"]
    }
    eval_index_paths = {
        name: os.path.join(eval_base_path, name.lower())
        for name in [ "hadith", "quran"]
    }

    all_results = {}

    for name, path in index_paths.items():
        if not os.path.exists(path):
            logging.warning(f"Index path not found: {path}")
            continue

        eval_index_path = eval_index_paths.get(name)

        logging.info(f"Running benchmark on index: {name}")
        
        try:
            model_faiss_index = VectorSearchDeployment(path, model_name, device)
            eval_faiss_index = load_precomputed_embeddings(eval_index_path, eval_model_name, device)
            eval_embed_func = eval_faiss_index.embeddings.embed_query
        except Exception as e:
            logging.error(f"Failed to initialize model or eval index for {name}: {e}")
            continue

        detailed_results = []
        for entry in tqdm(benchmark_data[:num_rows] if num_rows else benchmark_data, desc=f"Processing {name}"):
            result = process_entry(entry, name, model_faiss_index, method, k, sim_threshold, eval_embed_func, eval_faiss_index)
            if result is not None:
                detailed_results.append(result)

        recalls = [r["recall"] for r in detailed_results]
        precisions = [r["precision"] for r in detailed_results]

        avg_recall = np.mean(recalls) if recalls else 0.0
        avg_precision = np.mean(precisions) if precisions else 0.0
        
        print(f"Average Recall: {avg_recall:.2f}")
        print(f"Average Precision: {avg_precision:.2f}")

        all_results[name] = {
            "average_recall": float(avg_recall),
            "average_precision": float(avg_precision),
            "detailed_results": detailed_results
        }

    save_results(all_results, model_name, doctype, device)

def run_benchmark_wrapper(args):
    """Wrapper function to unpack arguments for multiprocessing"""
    model_name, doctype, device, benchmark_path, method = args
    return run_benchmark(model_name, doctype, device, benchmark_path, method)

if __name__ == '__main__':
    model_names = [
        "nomic-ai/nomic-embed-text-v1",
        "nomic-ai/nomic-embed-text-v2-moe",
        "Alibaba-NLP/gte-multilingual-base",
        "fine_tuned_models/islamqa_fine_tuned_all-mpnet-base-v2",
        "sentence-transformers/all-mpnet-base-v2",
        "sentence-transformers/LaBSE",
        "intfloat/multilingual-e5-base",
        'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
    ]

    device = "cpu"
    benchmark_path = "datasets/islamqa_references_benchmark.json"
    
    for model_name in model_names:
        method = "best_match_dedup"
        doctypes = ["preprocessed", "original"]

        # run_benchmark(model_name, doctypes[0], device, benchmark_path, method)

        with multiprocessing.Pool(processes=2) as pool:
            tasks = [(model_name, doctype, device, benchmark_path, method) for doctype in doctypes]
            results = pool.map(run_benchmark_wrapper, tasks)