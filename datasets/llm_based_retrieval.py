
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
# - If you cannot find relevant references that satisfies 10 for quran and 10 for hadith, do not try to hallucinate to make them; it is fine to have less than 20 items, but you must return most relevant references.

# Prompt template
def build_prompt(question):
    return f"""
You are a knowledgeable Islamic scholar and AI assistant. Given the following question, return a JSON object with the following structure:

[
    {{
    "type": "Quran" or "Hadith",
    "text": "<text of the ayah or hadith>",
    "reference": "<source reference, e.g., Al-Baqarah 2:2 or Muslim Book 1, Hadith 12>"
    }},
    ...
    // Total 20 items: 10 from Quran, next 10 from Hadith
]

Make sure:
- All references are accurate.
- Do not hallucinate references.
- Include only the most relevant references.
- The more references you can find without duplicates, the better. But keep the total under 20.
- If you cannot find 10 Quranic verses or 10 Hadiths, return as many as you can and fill up the rest with less relevant but related references.
- Do not include any duplicate references or other information or commentary.
- Only include authentic Hadiths from the six books (from Sahih Bukhari, Sahih Muslim, Tirmidhi, etc.).
- Format the JSON properly so it can be parsed by code.

Here is the question:
\"\"\"{question}\"\"\"

"""

# Function to run the chat completion and extract references
def extract_refs(question, model="gpt-4o"):
    cache_key = f"{question}"
    if cache_key in cache:
        logger.info(f"Retrieved from cache: {question}")
        return cache[cache_key]

    prompt = build_prompt(question)
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
        openai_response = extract_refs(question)
        results.append({
            "question": question,
            "openai_response_for_references": openai_response
        })
        logger.info(f"Processed question: {question}")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Results saved to {output_file}")

process_and_save(test_ds, output_file="openai_references_for_islamqa_benchmark.json")
