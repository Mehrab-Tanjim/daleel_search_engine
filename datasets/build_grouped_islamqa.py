"""Group reviewed query variants and export graded verse evidence."""
import hashlib
import json
from pathlib import Path
import random
import re
import sqlite3
import unicodedata

ROOT = Path(__file__).resolve().parent / 'sanitized_islamqa'
GRADES = {'direct': 3, 'partial': 2, 'background': 1, 'unrelated': 0}


def key(text):
    return ' '.join(re.findall(r'\w+', unicodedata.normalize('NFKC', text).casefold()))


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf8').splitlines()]


def write_lines(path, rows):
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf8')


def build(database):
    questions = json.loads((ROOT / 'group_questions.json').read_text(encoding='utf8'))
    candidates = json.loads((ROOT / 'similar_question_candidates.json').read_text(encoding='utf8'))
    reviews = [row for shard in range(3) for row in json.loads((ROOT / f'group_review_{shard}.json').read_text(encoding='utf8'))]
    decisions = {row['index']: row for row in reviews}
    assert len(decisions) == len(reviews) == len(candidates)
    assert set(decisions) == set(range(len(candidates)))
    assert all(isinstance(row['merge'], bool) and row['reason'].strip() for row in reviews)
    overrides = json.loads((ROOT / 'group_review_overrides.json').read_text(encoding='utf8'))
    for row in overrides:
        decisions[row['index']] = row
    approved = {tuple(sorted((candidates[i]['a'], candidates[i]['b']))) for i, row in decisions.items() if row['merge']}
    groups = {i: {i} for i in range(len(questions))}
    owner = list(range(len(questions)))
    # Require approval for every cross-group pair. Similarity chains cannot silently merge intents.
    for edge in sorted(candidates, key=lambda row: -row['cosine']):
        a, b = owner[edge['a']], owner[edge['b']]
        if a == b:
            continue
        if all(tuple(sorted((x,y))) in approved for x in groups[a] for y in groups[b]):
            for member in groups[b]:
                owner[member] = a
            groups[a].update(groups.pop(b))
    indexed = {key(question): i for i, question in enumerate(questions)}
    records = read_lines(ROOT / 'all_pairs_reviewed.jsonl')
    by_group = {group: [] for group in groups}
    for record in records:
        if record['review']['relevance'] != 'unresolved':
            by_group[owner[indexed[key(record['question'])]]].append(record)
    with sqlite3.connect(database) as conn:
        corpus = {f'{s}:{v}': text for s,v,text in conn.execute('select surah,verse,en from docs')}
    output = []
    conflicts = []
    qrels = []
    for group, members in sorted(groups.items()):
        source = by_group[group]
        variants = list(dict.fromkeys(record['question'] for record in source))
        group_id = hashlib.sha256('\n'.join(sorted(key(q) for q in variants)).encode('utf8')).hexdigest()[:16]
        evidence = {}
        dropped = []
        for record in source:
            label = record['review']['relevance']
            if not record['ayahs']:
                dropped.append(record['id'])
            for ref, text in zip(record['ayahs'], record['verse_texts']):
                evidence.setdefault(ref, {'ayah': ref, 'text': text, 'judgments': []})['judgments'].append({
                    'source_id': record['id'], 'query': record['question'], 'label': label,
                    'grade': GRADES[label], 'reason': record['review']['reason'],
                    'source_reference': record['source_reference'], 'reviewed_passage': record['ayahs'],
                })
        tiers = {label: [] for label in GRADES}
        for ref, item in evidence.items():
            labels = {judgment['label'] for judgment in item['judgments']}
            label = max(labels, key=GRADES.get)
            item['grade'] = GRADES[label]
            item['has_conflicting_grades'] = len(labels) > 1
            if len(labels) > 1:
                conflicts.append({'group_id': group_id, 'ayah': ref, 'labels': sorted(labels), 'chosen': label})
            tiers[label].append(item)
            qrels.append({'group_id': group_id, 'ayah': ref, 'grade': GRADES[label], 'conflict': len(labels)>1})
        sampled = []
        if not tiers['unrelated']:
            # Random examples are unjudged, not established irrelevant or hard negatives.
            excluded = set(evidence)
            for ref in evidence:
                surah, verse = map(int, ref.split(':'))
                excluded.update(f'{surah}:{near}' for near in range(max(1,verse-2),verse+3))
            options = sorted(set(corpus) - excluded)
            rng = random.Random(group_id)
            sampled = [{'ayah': ref, 'text': corpus[ref], 'review': 'random_unjudged', 'grade': None}
                       for ref in rng.sample(options, min(3,len(options)))]
        output.append({'id': group_id, 'queries': variants, 'source_splits': sorted({r['split'] for r in source}),
                       'relevance_order': ['direct','partial','background'], 'ayat': tiers,
                       'random_negative_candidates': sampled, 'missing_reference_records': dropped,
                       'training_eligible': any(tiers[label] for label in ('direct','partial','background'))})
    write_lines(ROOT / 'grouped_queries.jsonl', output)
    write_lines(ROOT / 'grouped_qrels.jsonl', qrels)
    write_lines(ROOT / 'grouped_conflicts.jsonl', conflicts)
    summary = {'source_pairs': len(records), 'excluded_unresolved': sum(r['review']['relevance']=='unresolved' for r in records),
               'groups': len(output), 'query_wordings': sum(len(r['queries']) for r in output),
               'multi_variant_groups': sum(len(r['queries'])>1 for r in output),
               'judged_query_verse_pairs': len(qrels), 'conflicting_query_verses': len(conflicts),
               'groups_by_tier': {label: sum(bool(r['ayat'][label]) for r in output) for label in GRADES},
               'groups_with_more_than_five_direct_verses': sum(len(r['ayat']['direct'])>5 for r in output),
               'cross_source_split_groups': sum(len(r['source_splits'])>1 for r in output)}
    (ROOT / 'grouped_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf8')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', required=True)
    build(parser.parse_args().corpus)
