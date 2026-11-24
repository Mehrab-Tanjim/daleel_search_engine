"""Validate v3 hashes, frozen splits and source-family training exclusion."""
import json
import sys
from pathlib import Path
from build_v3 import ROOT, V2, V3, read, sha

sys.path.insert(0,str(ROOT.parents[1]/'evaluation'))
from audit_islamqa_leakage import audit

def validate():
    manifest=json.loads((V3/'manifest.json').read_text())
    assert manifest['dataset_version']=='3.0.0'
    assert manifest['parent_manifest_sha256']==sha((V2/'manifest.json').read_bytes())
    for name,entry in manifest['outputs'].items():
        assert sha((V3/name).read_bytes())==entry['sha256'],name
        assert len(read(V3/name))==entry['records'],name
    for split in ('validation','test'):
        for name in ('questions.jsonl','grades.jsonl','unjudged_candidates.jsonl'):
            assert (V3/split/name).read_bytes()==(V2/split/name).read_bytes(),(split,name)
    questions={s:read(V3/s/'questions.jsonl') for s in ('train','validation','test')}
    removed={r['group_id'] for r in read(V3/'removed_training_groups.jsonl')}
    for name in ('questions.jsonl','grades.jsonl','unjudged_candidates.jsonl'):
        assert read(V3/'train'/name)==[r for r in read(V2/'train'/name) if r['group_id'] not in removed],name
    sources={s:json.loads((ROOT.parent/f'islamqa_references_{s}.json').read_text(encoding='utf8')) for s in ('train','benchmark')}
    report=audit(questions,sources,{'test_set_blind':False})
    assert not report['invalid_provenance']
    for entries in report['cross_split_identity_details'].values():
        assert not any('train' in x['splits'] for x in entries)
    assert not any('train' in x['splits'] for x in report['source_answer_overlaps'])
    print(json.dumps({'valid':True,'train_groups':len(questions['train']),'removed':len(removed),'validation_test_byte_identical':True,'train_source_overlaps':0,'retained_validation_test_answer_families':report['source_answer_overlap_count'],'test_blind':False},indent=2))

if __name__=='__main__':validate()
