from datasets import load_dataset
import os
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
import re
import json
import logging
from typing import List, Dict
from tqdm import tqdm 
from datasets import Dataset
from dotenv import load_dotenv  
import os
import json

def load_local_dataset(directory, train_questions):
    train_questions = set([t.strip().lower() for t in train_questions])  # Convert to set for faster lookup
    questions = []
    full_answers = []
    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".json"):  # Process only JSON files
            file_path = os.path.join(directory, filename)
            print(f"Reading file: {filename}")
            
            # Open and read the JSON file
            with open(file_path, "r", encoding="utf-8") as file:
                try:
                    data = json.load(file)
                    # Iterate through the list of dictionaries
                    for entry in data:
                        question = entry.get("question", "")
                        answer = entry.get("answer", "")

                        if question and question.strip().lower() not in train_questions:
                            questions.append(question)
                            full_answers.append(answer)
                        else:
                            print(f"Skipping question: {question} (already in train questions)")
                except json.JSONDecodeError as e:
                    print(f"Error reading {filename}: {e}")

    dataset_dict = {
        "Question": questions,
        "Full Answer": full_answers
    }

    benchmark_dataset = Dataset.from_dict(dataset_dict)
    return benchmark_dataset

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

# Load the IslamQA dataset from Hugging Face as Train
train_ds = load_dataset("minhalvp/islamqa", split="train")
train_ds_query = train_ds.filter(lambda x: x["Question"] != "")

# Load the IslamQA dataset from Kaggle as Test
test_ds = load_local_dataset("IslamQA/Kaggle", train_ds_query["Question"])


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
def process_and_save(dataset, output_file):
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

process_and_save(train_ds, output_file="islamqa_references_train.json")
process_and_save(test_ds, output_file="islamqa_references_benchmarks.json")
