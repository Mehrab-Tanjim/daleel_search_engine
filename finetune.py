import os
import json
from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments, losses
from sentence_transformers.evaluation import RerankingEvaluator, SimilarityFunction
import logging
from datetime import datetime
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np
import torch
import random
from sklearn.metrics.pairwise import cosine_similarity

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def chunk_text(text, tokenizer, max_length=512, overlap=50):
    """Chunk text into smaller pieces fitting within max_length tokens"""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunk_size = max_length - overlap
    chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size - overlap)]
    text_chunks = [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks if chunk]
    return [chunk for chunk in text_chunks if chunk.strip()]

def load_data(dataset_path, tokenizer, max_length=512):
    """Load JSON data and prepare for training and evaluation"""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    anchors = []
    positives = []
    all_answers = []  # For sampling negatives
    
    for entry in tqdm(data, desc="Processing entries"):
        question = entry.get("question", "").strip()
        full_answer = entry.get("answer", "").strip()
        
        entry_positives = []

        # TODO: optionally use the full answer but for our use-case we only need the references
        # if full_answer:
        #     answer_chunks = chunk_text(full_answer, tokenizer, max_length)
        #     entry_positives.extend(answer_chunks)
        #     all_answers.extend(answer_chunks)
        
        for ref in entry.get("extracted_references", []):
            ref_text = ref.get("text", "").strip()
            if question and ref_text and question != ref_text:
                entry_positives.append(ref_text)
                all_answers.append(ref_text)
        
        for pos in entry_positives:
            anchors.append(question)
            positives.append(pos)
    
    if not anchors or not positives:
        raise ValueError("No valid examples found in dataset")
    
    dataset_dict = {"anchor": anchors, "positive": positives}
    full_dataset = Dataset.from_dict(dataset_dict)
    logging.info(f"Loaded {len(full_dataset)} anchor-positive pairs")
    
    return full_dataset, all_answers

def prepare_validation_dataset(dataset, all_answers, train_test_split_ratio=0.8):
    """Split dataset and prepare validation examples"""
    split_dataset = dataset.train_test_split(train_size=train_test_split_ratio, seed=42)
    train_data = split_dataset["train"]
    val_data = split_dataset["test"]
    
    samples = []
    for item in tqdm(val_data):
        query = item["anchor"].strip()
        positive = item["positive"].strip()
        
        # Sample negatives from all_answers, excluding positives for this query
        # Please note that this is not hard negative sampling, but a simple random sampling
        possible_negatives = [a for a in all_answers if a != positive and a != query]
        negatives = random.sample(possible_negatives, min(3, len(possible_negatives))) if possible_negatives else []
        
        if query and positive:
            samples.append({
                "query": query,
                "positive": [positive],
                "negative": negatives
            })

    if not samples:
        raise ValueError("No valid reranking samples created")
    
    logging.info(f"Prepared {len(samples)} reranking samples (queries: {len(samples)}, positives: {sum(len(s['positive']) for s in samples)}, negatives: {sum(len(s['negative']) for s in samples)})")
    return split_dataset["train"], samples


def compute_cosine_similarity_score(model, val_pairs, batch_size=8, device="cuda"):
    """Compute average cosine similarity between anchor-positive pairs"""
    model.eval()
    with torch.no_grad():
        
        embeddings1 = model.encode([x[0] for x in val_pairs], convert_to_tensor=True, device=device)
        embeddings2 = model.encode([x[1] for x in val_pairs], convert_to_tensor=True, device=device)
        
        cosine_scores = torch.nn.functional.cosine_similarity(embeddings1, embeddings2)

    return cosine_scores.mean().item()

def fine_tune_model(model_name, dataset_path, output_dir, max_length=512, num_epochs=3, batch_size=8, learning_rate=5e-5, weight_decay=0.01):
    """Fine-tune using SentenceTransformerTrainer with custom cosine similarity evaluator"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # Load model and tokenizer
    model = SentenceTransformer(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.to(device)

    # Load and preprocess dataset
    full_dataset, all_answers = load_data(dataset_path, tokenizer, max_length)
    train_dataset, reranking_samples = prepare_validation_dataset(full_dataset, all_answers)

    # Define loss function
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # Define evaluator (custom cosine similarity evaluator)
    # TODO: For a simple cosine similarity evaluation, we will need to create a custom evaluator class
    # based on the RerankingEvaluator class from sentence-transformers
    # def evaluator(model):
    #     return compute_cosine_similarity_score(model, val_examples, batch_size)

     # Define evaluator
    evaluator = RerankingEvaluator(
        samples=reranking_samples,
        name="islamqa-reranking",
    )
    # Initial evaluation
    initial_score = evaluator(model)
    logging.info(f"Initial Evaluation Score: {initial_score}")

    
    # Set up training arguments
    training_args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=100,
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,  # Keep only the best model
        load_best_model_at_end=True,
        metric_for_best_model="eval_islamqa-reranking_ndcg@10",
        # logging_steps=50,
        # max_grad_norm=1.0,  # Gradient clipping
    )

    # Set up trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=None,  # We evaluate manually using the custom evaluator
        loss=train_loss,
        evaluator=evaluator,
        args=training_args  # Pass the training arguments here
    )

    # Train the model
    logging.info(f"Starting fine-tuning for model: {model_name}")
    trainer.train()

    # Final evaluation
    final_score = evaluator(model)
    logging.info(f"Final Evaluation Score: {final_score}")

    # Save the final model
    final_model_path = os.path.join(output_dir, "final_model")
    model.save(final_model_path)
    logging.info(f"Final model saved to {final_model_path}")

    # Clean up
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

if __name__ == "__main__":
    dataset_path = "datasets/islamqa_references_train.json"
    model_names = ["sentence-transformers/all-mpnet-base-v2"]
    
    for model_name in model_names:
        output_dir = f"fine_tuned_models/islamqa_fine_tuned_{model_name.split('/')[-1]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(output_dir, exist_ok=True)
        fine_tune_model(model_name, dataset_path, output_dir)