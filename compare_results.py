import json
from collections import defaultdict
import os
def compare_json_files(file1, file2):
    # Load JSON data from files
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        data1 = json.load(f1)
        data2 = json.load(f2)
    data1_dict_list = data1['hadith']['detailed_results']
    data2_dict_list = data2['hadith']['detailed_results']
    # Iterate through the JSON objects
    for dict1, dict2 in zip(data1_dict_list, data2_dict_list):
        if dict1["query"].replace('search_query: ', '') == dict2["query"].replace('search_query: ', ''):
            if dict1["matched"] == 1 and dict2["matched"] == 0:
    
                print(dict1["query"])
                print(dict1["ground_truths"])
                # print(dict1["retrieved_text"])
                # print(dict2["retrieved_text"])
                print("*******************************")

if __name__ == "__main__":

        
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

    eval_model_names = ["nomic-ai/nomic-embed-text-v1", "sentence-transformers/all-mpnet-base-v2", "Alibaba-NLP/gte-multilingual-base"]

    doctypes = ["original", "preprocessed"]
    device = "cpu" # "cuda"
    output_dir = "evaluation_results"
    ranking = defaultdict(list)
    
    for eval_model_name in eval_model_names:
        for model_name in model_names:
            for doctype in doctypes:
                result_file = f"{output_dir}/eval_model_{eval_model_name.split('/')[-1]}/{model_name.split('/')[-1]}_{doctype}_{device}.json"

                with open(result_file, 'r') as f:
                    results = json.load(f)
                # Get the ranking for the model
                ranking[eval_model_name].append((model_name+'_'+doctype, results['hadith']['average_recall']))
        # Sort the ranking based on average recall
        ranking[eval_model_name].sort(key=lambda x: x[1], reverse=True)
        # Print the ranking for each model
        # print(f"Ranking for {eval_model_name}: {[x[0] for x in ranking[eval_model_name]]}")    

    # find the common models in the same index for all eval model in ranking, in ordered by the first eval model
    rankings = [ranking[eval_model_name] for eval_model_name in eval_model_names]

    common_at_same_index = []

    # Iterate over indices
    for i in range(len(rankings[0])):
        # Get the item at this index in each list
        items_at_index = [ranking[i] for ranking in rankings]
        # If all are the same, add to result
        if len(set([item[0] for item in items_at_index])) == 1:
            common_at_same_index.append((i, items_at_index[0][0]))

    # Display results
    for index, name in common_at_same_index:
        print(f"Common at index {index}: {name}")

    # Compare two best models
    file1 = 'evaluation_results/eval_model_gte-multilingual-base/islamqa_fine_tuned_all-mpnet-base-v2_preprocessed_cpu.json'
    file2 = 'evaluation_results/eval_model_gte-multilingual-base/nomic-embed-text-v2-moe_preprocessed_cpu.json'

    compare_json_files(file1, file2)