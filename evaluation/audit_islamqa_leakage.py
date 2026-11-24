"""Audit source-family overlap without changing the published split package.

Exit 1 on cross-split identity/source-answer overlap or invalid provenance.
--require-blind also fails: repository metadata cannot certify private custody.
This supplements validate_dataset.py; it is not a semantic-equivalence oracle.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ('train', 'validation', 'test')


def normalize(text):
    return ' '.join(re.findall(r'\w+', unicodedata.normalize('NFKC', text).casefold()))


def sha256(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def audit(questions, sources, annotation=None):
    identities = {name: defaultdict(set) for name in
                  ('group_id', 'component_id', 'query_id', 'normalized_question', 'source_reference_id', 'source_article_row')}
    families = defaultdict(lambda: {'splits': set(), 'groups': set(), 'articles': set(), 'raw_answer_hashes': set()})
    problems = []
    split_for_group = {}
    for split in SPLITS:
        for question in questions[split]:
            gid = question['group_id']
            split_for_group[gid] = split
            identities['group_id'][gid].add(split)
            identities['component_id'][question['component_id']].add(split)
            queries = {q['query_id']: q['text'] for q in question['queries']}
            for qid, text in queries.items():
                identities['query_id'][qid].add(split)
                identities['normalized_question'][sha256(normalize(text))].add(split)
            for verses in question['evidence'].values():
                for verse in verses:
                    for judgment in verse['judgments']:
                        sid = judgment['source_id']
                        identities['source_reference_id'][sid].add(split)
                        try:
                            origin, row, reference = sid.split(':')
                            row, reference = int(row), int(reference)
                            if row < 0 or reference < 0:
                                raise ValueError('negative source index')
                            source = sources[origin][row]
                            source['extracted_references'][reference]
                            text = judgment.get('source_query', queries[judgment['query_id']])
                            if normalize(source['question']) != normalize(text):
                                raise ValueError('source question differs from judgment query')
                        except (KeyError, IndexError, TypeError, ValueError) as error:
                            problems.append({'group_id': gid, 'source_id': sid, 'error': str(error)})
                            continue
                        article = f'{origin}:{row}'
                        identities['source_article_row'][article].add(split)
                        answer = source.get('answer', '')
                        normalized_answer = normalize(answer)
                        if not normalized_answer:
                            problems.append({'group_id': gid, 'source_id': sid, 'error': 'missing source answer'})
                            continue
                        family = families[sha256(normalized_answer)]
                        family['splits'].add(split)
                        family['groups'].add(gid)
                        family['articles'].add(article)
                        family['raw_answer_hashes'].add(sha256(answer))
    overlaps = {
        kind: [{'key': key, 'splits': sorted(splits)} for key, splits in sorted(index.items()) if len(splits) > 1]
        for kind, index in identities.items()
    }
    answer_overlaps = [
        {'normalized_answer_sha256': key, **{k: sorted(v) for k,v in family.items()},
         'raw_answers_identical': len(family['raw_answer_hashes']) == 1}
        for key,family in sorted(families.items()) if len(family['splits']) > 1
    ]
    affected = {gid for family in answer_overlaps for gid in family['groups']}
    legacy = {}
    for origin, source_rows in sources.items():
        known = {normalize(row['question']) for row in source_rows}
        legacy[origin] = {
            split: sum(any(normalize(q['text']) in known for q in row['queries']) for row in questions[split])
            for split in SPLITS
        }
    passed = not problems and not answer_overlaps and not any(overlaps.values())
    return {
        'structural_source_checks_pass': passed,
        'cross_split_identity_counts': {k: len(v) for k,v in overlaps.items()},
        'cross_split_identity_details': overlaps,
        'source_answer_overlap_count': len(answer_overlaps),
        'byte_identical_answer_overlap_count': sum(f['raw_answers_identical'] for f in answer_overlaps),
        'source_answer_overlap_by_splits': dict(sorted(Counter('|'.join(f['splits']) for f in answer_overlaps).items())),
        'affected_groups_by_split': dict(sorted(Counter(split_for_group[gid] for gid in affected).items())),
        'source_answer_overlaps': answer_overlaps,
        'invalid_provenance': problems,
        'legacy_source_question_matches_by_group': legacy,
        'legacy_exposure_note': 'Source membership is not proof that any particular checkpoint was trained on it. Audit checkpoint lineage separately.',
        'blindness': {
            'manifest_claim': (annotation or {}).get('test_set_blind'),
            'certified': False,
            'status': 'not_blind' if (annotation or {}).get('test_set_blind') is False else 'unverified',
            'reason': 'A manifest flag or lack of exact overlap cannot establish hidden custody, no prior exposure, or no pretraining contamination.',
        },
        'limits': [
            'Source row IDs are local to each original input; exact full-answer fingerprints catch duplicate articles under different rows/titles.',
            'Source URL lineage, near duplicates and compound/shared-intent families need additional review.',
            'The same target ayah across splits is expected for fixed-corpus retrieval and is not itself leakage.',
            'This report contains public source identifiers/hashes, not question or answer text.',
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, default=ROOT/'datasets/sanitized_islamqa/split_package')
    parser.add_argument('--sources', type=Path, default=ROOT/'datasets')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--require-blind', action='store_true')
    args = parser.parse_args()
    questions = {s: read_jsonl(args.package/s/'questions.jsonl') for s in SPLITS}
    sources = {s: json.loads((args.sources/f'islamqa_references_{s}.json').read_text(encoding='utf-8')) for s in ('train','benchmark')}
    manifest = json.loads((args.package/'manifest.json').read_text(encoding='utf-8'))
    report = audit(questions, sources, manifest.get('annotation'))
    inputs = [args.package/'manifest.json'] + [args.package/s/'questions.jsonl' for s in SPLITS] + [args.sources/f'islamqa_references_{s}.json' for s in sources]
    report['input_sha256'] = {str(p.relative_to(args.package)) if p.is_relative_to(args.package) else p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    report['gate_pass'] = report['structural_source_checks_pass'] and not args.require_blind
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('gate_pass','cross_split_identity_counts','source_answer_overlap_count','byte_identical_answer_overlap_count','source_answer_overlap_by_splits','affected_groups_by_split','legacy_source_question_matches_by_group','blindness')},indent=2))
    return 0 if report['gate_pass'] else 1


if __name__ == '__main__':
    sys.exit(main())
