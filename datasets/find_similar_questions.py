"""Find E5 cosine > 0.90 candidates for manual question grouping."""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

ROOT = Path(__file__).resolve().parent / 'sanitized_islamqa'
MODEL = 'intfloat/multilingual-e5-small'
REVISION = 'c007d7ef6fd86656326059b28395a7a03a7c5846'


def main():
    torch.set_num_threads(4)
    questions = json.loads((ROOT / 'group_questions.json').read_text(encoding='utf8'))
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    model = AutoModel.from_pretrained(MODEL, revision=REVISION, local_files_only=True).eval()
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(questions), 16):
            tokens = tokenizer(['query: ' + q for q in questions[start:start+16]],
                               padding=True, truncation=True, max_length=512, return_tensors='pt')
            hidden = model(**tokens).last_hidden_state
            mask = tokens.attention_mask.unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1)
            vectors.append(torch.nn.functional.normalize(pooled, dim=1).numpy())
            if start % 160 == 0:
                print(start, '/', len(questions), flush=True)
    embeddings = np.concatenate(vectors)
    matrix = embeddings @ embeddings.T
    candidates = []
    for a, b in zip(*np.where(np.triu(matrix > 0.90, 1))):
        candidates.append({'a': int(a), 'b': int(b), 'cosine': float(matrix[a,b]),
                           'question_a': questions[a], 'question_b': questions[b]})
    (ROOT / 'similar_question_candidates.json').write_text(json.dumps(candidates, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print('candidates', len(candidates), flush=True)


if __name__ == '__main__':
    main()
