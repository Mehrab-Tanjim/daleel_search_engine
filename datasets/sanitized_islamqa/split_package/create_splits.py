"""Rebuild the provisional group-safe splits from the reviewed dataset."""
import collections
import hashlib
import json
from pathlib import Path
import random
import re
import unicodedata

PACKAGE = Path(__file__).resolve().parent
DATA = PACKAGE.parent


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf8').splitlines()]


def write_lines(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in records), encoding='utf8')


def normalize(text):
    return ' '.join(re.findall(r'\w+', unicodedata.normalize('NFKC', text).casefold()))


def main():
    groups = read_lines(DATA / 'grouped_queries.jsonl')
    questions = json.loads((DATA / 'group_questions.json').read_text(encoding='utf8'))
    candidates = json.loads((DATA / 'similar_question_candidates.json').read_text(encoding='utf8'))
    conflicts = read_lines(DATA / 'grouped_conflicts.jsonl')
    question_index = {normalize(question): i for i, question in enumerate(questions)}
    parent = list(range(len(questions)))

    def root(index):
        while index != parent[index]:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for candidate in candidates:
        assert candidate['cosine'] > 0.90
        parent[root(candidate['a'])] = root(candidate['b'])
    clusters = collections.defaultdict(list)
    for group in groups:
        roots = {root(question_index[normalize(query)]) for query in group['queries']}
        assert len(roots) == 1
        clusters[next(iter(roots))].append(group)

    components = list(clusters.values())
    random.Random(4213).shuffle(components)
    components.sort(key=len, reverse=True)
    ratios = {'train': .65, 'validation': .15, 'test': .20}
    splits = {name: [] for name in ratios}
    targets = {name: len(groups) * ratio for name, ratio in ratios.items()}
    for component in components:
        name = max(splits, key=lambda item: (targets[item] - len(splits[item])) / targets[item])
        splits[name].extend(component)

    excluded = {(row['group_id'], row['ayah']) for row in conflicts}
    ownership = {}
    summary = {}
    for name, items in splits.items():
        for item in items:
            assert item['id'] not in ownership
            ownership[item['id']] = name
        grades = [{'group_id': group['id'], 'ayah': verse['ayah'], 'grade': verse['grade']}
                  for group in items for verses in group['ayat'].values() for verse in verses
                  if (group['id'], verse['ayah']) not in excluded]
        write_lines(PACKAGE / name / 'questions.jsonl', items)
        write_lines(PACKAGE / name / 'grades.jsonl', grades)
        summary[name] = {'question_groups': len(items), 'grades': len(grades),
                         'direct_question_groups': sum(any(v['grade'] == 3 and (g['id'], v['ayah']) not in excluded
                                                       for v in g['ayat']['direct']) for g in items)}
    assert len(ownership) == len(groups)
    for component in components:
        assert len({ownership[group['id']] for group in component}) == 1
    write_lines(PACKAGE / 'conflicts_excluded_from_grades.jsonl', conflicts)
    summary['source_sha256'] = {name: hashlib.sha256((DATA / name).read_bytes()).hexdigest() for name in
                                ['grouped_queries.jsonl', 'similar_question_candidates.json', 'grouped_conflicts.jsonl']}
    summary['similarity_threshold'] = '> 0.90'
    summary['random_seed'] = 4213
    summary['largest_similarity_component'] = max(map(len, components))
    summary['excluded_conflicting_grades'] = len(excluded)
    (PACKAGE / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
