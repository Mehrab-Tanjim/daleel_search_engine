import pickle
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain.docstore.document import Document
from functools import lru_cache
from tqdm import tqdm
import torch

# Initialize LongT5 tokenizer and model globally (to avoid reloading per call)
model_name = "google/long-t5-tglobal-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    device_map="auto"  # Automatically use GPU if available
)
if torch.cuda.is_available():
    model = model.to("cuda")  # Ensure model is on GPU if available

def preprocess_text(retrived_text):
    retrived_text = retrived_text[retrived_text.find(":") + 1:].strip()
    retrived_text = retrived_text.replace('“', '"').replace('”', '"').replace("‘", "'").replace("’", "'").replace("\'", "").replace('`','').replace('--','').replace('[??]','')
    retrived_text = re.sub(r'\([^)]*\)', '', retrived_text)
    retrived_text = re.sub(r'\s+([.,!?])', r'\1', retrived_text)
    retrived_text = re.sub(r'\s+', ' ', retrived_text)
    retrived_text = re.sub(r'([.?!])\1+', r'\1', retrived_text)
    return retrived_text

@lru_cache(maxsize=10000)
def summarize_if_needed(text: str, threshold=100) -> str:
    if len(text.split()) > threshold:
        try:
            # Use improved LongT5 summarization logic
            inputs = tokenizer.encode(
                "Summarize: " + text,
                return_tensors='pt',
                max_length=16384,  # LongT5 supports up to 16k tokens
                truncation=True
            )
            if torch.cuda.is_available():
                inputs = inputs.to("cuda")  # Move inputs to GPU if available

            output = model.generate(
                    inputs,
                    max_length=150,  # Increase to allow more detail
                    min_length=100,  # Slightly longer minimum for better coverage
                    # num_beams=8,  # Higher beams for improved coherence and quality
                    # length_penalty=0.5,  # Slightly favor shorter, denser summaries
                    # no_repeat_ngram_size=3,  # Stronger repetition prevention
                    early_stopping=True  # Stop when complete
                )
            
            summary = tokenizer.decode(output[0], skip_special_tokens=True)
            torch.cuda.empty_cache()  # Clear GPU memory after generation
            return summary.strip()
        except Exception as e:
            print("Summarization error:", e)
            return ""
    return ""

def preprocess_documents(docs, source='hadith'):
    prepared = []
    for doc in tqdm(docs):
        original = doc.page_content
        cleaned = preprocess_text(original)
        summary_text = "" #summarize_if_needed(original) if source == "hadith" else ""
        new_doc = Document(page_content=cleaned, metadata={**doc.metadata, "original_text": original, "summary_text": summary_text})
        prepared.append(new_doc)
    return prepared

# Load original raw documents
with open("datasets/original_quran_docs.pkl", "rb") as f:
    original_quran_docs = pickle.load(f)
with open("datasets/original_hadith_docs.pkl", "rb") as f:
    original_hadith_docs = pickle.load(f)

# Prepare and save processed documents
preprocessed_quran_docs = preprocess_documents(original_quran_docs, source='quran')
preprocessed_hadith_docs = preprocess_documents(original_hadith_docs, source='hadith')

with open("datasets/preprocessed_quran_docs.pkl", "wb") as f:
    pickle.dump(preprocessed_quran_docs, f)
with open("datasets/preprocessed_hadith_docs.pkl", "wb") as f:
    pickle.dump(preprocessed_hadith_docs, f)

print("✅ preprocessed documents saved.")