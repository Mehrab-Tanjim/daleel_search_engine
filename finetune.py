import os
import json
from datasets import Dataset
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import logging
from datetime import datetime
from datasets import load_dataset
from transformers import AutoTokenizer
import torch
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from tqdm import tqdm

def chunk_text(text, tokenizer, max_length=512, overlap=50):
    """Chunk text into smaller pieces fitting within max_length tokens"""
    
    # Tokenize the entire text
    tokens = tokenizer.encode(text, add_special_tokens=False)

    # Chunk into 512-token segments
    chunk_size = max_length
    chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]

    # Decode chunks back to text
    text_chunks = [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]
    
    return text_chunks

def load_data(dataset_path, tokenizer, max_length=512):
    """Load JSON data and convert to Dataset format"""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Prepare examples: question paired with each reference text
    examples = []
    for entry in tqdm(data, desc="Processing entries"):
        question = entry.get("question", "").strip()
        full_answer = entry.get("answer", "").strip()
        if full_answer:
            answer_chunks = chunk_text(full_answer, tokenizer, max_length)
            for chunk in answer_chunks:
                examples.append({"Question":question, "Answer":chunk})
        for ref in entry.get("extracted_references", []):
            ref_text = ref.get("text", "").strip()
            if question and ref_text:
                examples.append({"Question":question, "Answer":ref_text})
    
    return Dataset.from_list(examples)

def prepare_training_examples(dataset, train_test_split_ratio=0.8):
    """Prepare InputExamples for training and validation"""
    split_dataset = dataset.train_test_split(train_size=train_test_split_ratio, seed=42)
    
    train_data = split_dataset["train"]
    val_data = split_dataset["test"]

    train_examples = [InputExample(texts=[item["Question"].strip(), item["Answer"]]) 
                     for item in train_data]
    val_examples = [(item["Question"].strip(), item["Answer"]) 
                   for item in val_data]  # For evaluator
    
    return train_examples, val_examples

def fine_tune_model(model_name, dataset_path, output_dir, max_length=512, num_epochs=3, batch_size=8, learning_rate=5e-5, weight_decay=0.01):
    """Fine-tune a SentenceTransformer model for semantic retrieval with additional training args"""
    # Load model
    device = "cuda" 
    model = SentenceTransformer(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = model.to(device)  

    # Load and preprocess dataset
    dataset = load_data(dataset_path, tokenizer, max_length)
    train_examples, val_examples = prepare_training_examples(dataset)
    
    # Create DataLoader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    
    # Define loss function
    train_loss = losses.MultipleNegativesRankingLoss(model)
    
    # Define evaluator
    evaluator = EmbeddingSimilarityEvaluator(
        sentences1=[x[0] for x in val_examples],
        sentences2=[x[1] for x in val_examples],
        scores=[1.0] * len(val_examples),
        main_similarity="cosine",
        batch_size=batch_size  # Match per_device_eval_batch_size
    )


    # Calculate steps per epoch (approximation)
    steps_per_epoch = len(train_examples) // batch_size
    evaluation_steps = steps_per_epoch  # Evaluate at end of each epoch
    
    # Training with additional arguments
    logging.info(f"Starting fine-tuning for model: {model_name}")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=num_epochs,
        warmup_steps=100,
        output_path=output_dir,
        evaluation_steps=50, #evaluation_steps,  # Mimics evaluation_strategy="epoch"
        # optimizer_params={"lr": learning_rate, "weight_decay": weight_decay},  # Incorporates learning_rate and weight_decay
        show_progress_bar=True,
        save_best_model=True  # Mimics load_best_model_at_end=True
    )
    
    logging.info(f"Fine-tuned model saved to {output_dir}")
    del model  # Clear memory
    torch.cuda.empty_cache()  # Clear GPU memory if using CUDA
    # return model

if __name__ == "__main__":
    # Define paths and parameters
    dataset_path = "datasets/islamqa_references_train.json"
    model_names = ["Alibaba-NLP/gte-multilingual-base", "sentence-transformers/all-mpnet-base-v2"]
    
    for model_name in model_names:
        output_dir = f"fine_tuned_models/islamqa_fine_tuned_{model_name.split('/')[-1]}"
        os.makedirs(output_dir, exist_ok=True)
        
        # max_length = 512 if model_name != "Alibaba-NLP/gte-multilingual-base" else 8192
        # Fine-tune the model
        fine_tune_model(model_name, dataset_path, output_dir)#, max_length=max_length)