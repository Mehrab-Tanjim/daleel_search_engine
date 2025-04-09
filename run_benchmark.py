import json
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from utils import *
import re
from tqdm import tqdm
import logging
from datetime import datetime
import numpy as np
from multiprocessing import Pool
from functools import partial
import multiprocessing

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
    # TODO: this is not accurate at all, as it depends on the embedding model. For objective evaluator, we need to use the same model for both reference and result embeddings. Still this can be give a comparative score for the same model

    ref_embedding = embed_func(ref_text)
    
    # Cosine similarity
    dot_product = np.dot(result_embeddings, ref_embedding)
    norms = np.linalg.norm(ref_embedding) * np.linalg.norm(result_embeddings, axis=1)
    similarities = dot_product / (norms + 1e-8)  # Avoid division by zero
    return np.max(similarities)

def evaluate_retrieval(results, ground_truth_refs, result_embeddings = None, matching='exact', embed_func=None, sim_threshold=0.75):
    matched = 0
    retrieved = len(results) if results else 0
    total_refs = len(ground_truth_refs) if ground_truth_refs else 0

    # if not results or not ground_truth_refs:
    if not results or not ground_truth_refs:
        return 0, total_refs, retrieved, 0.0, 0.0

    if matching == 'exact':
        results_lower = set([r.lower() for r in results])
        for ref in ground_truth_refs:
            if ref.lower() in results_lower:
                matched += 1
    else:
        for ref in ground_truth_refs:
            sim_text = batch_semantic_match_score(ref[0], result_embeddings, embed_func)
            # sim_reference = batch_semantic_match_score(ref[1], [res[1] for res in results], embed_func)
            if sim_text >= sim_threshold: # or sim_reference >= sim_threshold
                matched += 1

    precision = matched / retrieved if retrieved > 0 else 0.0
    recall = matched / total_refs if total_refs > 0 else 0.0
    return matched, total_refs, retrieved, precision, recall

def save_results(results, model_name, doctype, device, output_dir="evaluation_results"):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/eval_{model_name.split('/')[-1]}_{doctype}_{device}.json" #_{timestamp}
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    logging.info(f"Results saved to {filename}")

def process_entry(entry, name, model, method, k, sim_threshold):
    
    
    query = entry["question"].strip()

    try:
        if name == 'quran':
        
            references = [extract_colon_reference(ref["reference"]) 
                     for ref in entry["extracted_references"] 
                     if ref['type'].lower() == name]
            
            if not references:
                return None
                
            retrieved_docs = model.search(method, query, k)
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

            retrieved_docs, retrieved_embeddings = model.search(method, query, k, return_embeddings=True)
            retrieved_texts = [(doc.page_content, 
                          ' '.join([doc.metadata['source'], 
                                  doc.metadata['chapter_no'], 
                                  doc.metadata['hadith_no']])) 
                         for doc, score in retrieved_docs]

            matched, expected, retrieved, precision, recall = evaluate_retrieval(
                retrieved_texts, references, retrieved_embeddings, matching='cosine', 
                embed_func=model.embeddings.embed_query, sim_threshold=sim_threshold)

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
        
        

def run_benchmark(model_name, doctype, device, benchmark_path, method, k=5, sim_threshold=0.75, num_rows=None):
    try:
        benchmark_data = load_benchmark_data(benchmark_path)
    except Exception as e:
        logging.error(f"Failed to load benchmark data: {e}")
        return

    base_path = f"vector_databases/{model_name.split('/')[-1]}_{doctype}_{device}"
    index_paths = {
        name: os.path.join(base_path, name.lower())
        for name in [ "quran", "hadith"]
    }

    all_results = {}

    for name, path in index_paths.items():
        if not os.path.exists(path):
            logging.warning(f"Index path not found: {path}")
            continue

        logging.info(f"Running benchmark on index: {name}")
        
        try:
            model = VectorSearchDeployment(path, model_name, device)
        except Exception as e:
            logging.error(f"Failed to initialize model for {name}: {e}")
            continue

        detailed_results = []
        for entry in tqdm(benchmark_data[:num_rows] if num_rows else benchmark_data, desc=f"Processing {name}"):
            result = process_entry(entry, name, model, method, k, sim_threshold)
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
        "Alibaba-NLP/gte-multilingual-base",
        "sentence-transformers/all-mpnet-base-v2",
        "sentence-transformers/LaBSE",
        "intfloat/multilingual-e5-base",
        'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
    ]

    device = "cpu"
    benchmark_path = "datasets/islamqa_references_train.json"
    
    for model_name in model_names:
        method = "best_match_dedup" #best_match_dedup
        doctypes = ["preprocessed", "original"]

        # Create a pool with 2 processes (one for each document type)
        with multiprocessing.Pool(processes=2) as pool:
            # Prepare arguments for each doctype
            tasks = [(model_name, doctype, device, benchmark_path, method) for doctype in doctypes]
            # Run the tasks in parallel
            results = pool.map(run_benchmark_wrapper, tasks)