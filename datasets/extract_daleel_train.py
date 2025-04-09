from datasets import load_dataset
import os
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
import re
import json
import logging
from typing import List, Dict
from tqdm import tqdm 

from dotenv import load_dotenv  

# Load environment variables from .env
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client with your API key
api_key = os.getenv("OPENAI_API_KEY") 
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")
client = OpenAI(api_key=api_key) 

# Load the IslamQA dataset
ds = load_dataset("minhalvp/islamqa", split="train")

# Optional: Select a few entries to try (adjust or remove for full dataset)
sample_data = ds.select(range(5))

# Cache to store results
cache = {}

# Prompt template
def build_prompt(question, answer):
    return f"""
You are an expert Islamic scholar.

Your task is to extract ONLY direct Quranic verses and Hadiths mentioned in the answer below. Do NOT include commentary, explanation, or fatwas.

Return a JSON array where each item is either a Quranic verse or a Hadith. Format each item like this:
- type: "Quran" or "Hadith"
- text: the actual quoted ayat or hadith
- reference: if available (e.g. Surah name and number, Hadith collection)

This is very important. Return only in JSON format.

Example:
[
  {{
    "type": "Quran",
    "text": "Indeed, prayer prohibits immorality and wrongdoing.",
    "reference": "Surah Al-Ankabut 29:45"
  }},
  {{
    "type": "Hadith",
    "text": "Actions are but by intention.",
    "reference": "Sahih Bukhari, Book 1, Hadith 1"
  }}
]

Now extract from the following Islamic Q&A pair:

Question: {question}

Answer: {answer}
"""

# Function to run the chat completion and extract references
def extract_refs(question, answer, model="gpt-4o"):
    cache_key = f"{question}_{answer}"
    if cache_key in cache:
        logger.info(f"Retrieved from cache: {question}")
        return cache[cache_key]

    prompt = build_prompt(question, answer)
    messages: List[ChatCompletionMessageParam] = [{"role": "user", "content": prompt}]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2  # Lower temperature for factual extraction
        )
        content = response.choices[0].message.content
        json_match = re.search(r"```json\s*([\s\S]*?)\s*```", content)
        if json_match:
            result = json.loads(json_match.group(1))
            cache[cache_key] = result
            return result
        else:
            logger.warning(f"No JSON found in response for question: {question}")
            return []
    except Exception as e:
        logger.error(f"Error processing question: {question}. Error: {str(e)}")
        return []

# Function to process the dataset and save results
def process_and_save(dataset, output_file="islamqa_references_train.json"):
    results = []
    for item in tqdm(dataset):
        question = item["Question"]
        answer = item["Full Answer"]
        extracted = extract_refs(question, answer)
        results.append({
            "question": question,
            "answer": answer,
            "extracted_references": extracted
        })
        logger.info(f"Processed question: {question}")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Results saved to {output_file}")

# Run on the sample data
# process_and_save(sample_data)

# To process the entire dataset, uncomment the following line:
process_and_save(ds)