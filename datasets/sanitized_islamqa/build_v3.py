"""Build v3 by removing train families touching frozen validation/test."""
import hashlib
import json
from pathlib import Path
import re
import unicodedata

ROOT = Path(__file__).resolve().parent
V2 = ROOT / 'split_package'
V3 = ROOT / 'split_package_v3'

def read(path):
    return [json.loads(x) for x in path.read_text(encoding='utf8').splitlines() if x.strip()]

def norm(text):
    return ' '.join(re.findall(r'\w+', unicodedata.normalize('NFKC',text).casefold()))

def sha(data):
    return hashlib.sha256(data).hexdigest()

def write(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows).encode())

def build():
    sources = {s:json.loads((ROOT.parent/f'islamqa_references_{s}.json').read_text(encoding='utf8')) for s in ('train','benchmark')}
    questions = {s:read(V2/s/'questions.jsonl') for s in ('train','validation','test')}
    parent = {}
    def find(x):
        parent.setdefault(x,x)
        if parent[x]!=x: parent[x]=find(parent[x])
        return parent[x]
    def union(a,b):
        parent[find(a)]=find(b)
    owners={}
    for split,rows in questions.items():
        for row in rows:
            gid=row['group_id']; find(gid)
            keys=[('component',row['component_id'])]+[('query',norm(q['text'])) for q in row['queries']]
            for verses in row['evidence'].values():
                for verse in verses:
                    for j in verse['judgments']:
                        origin,index,_=j['source_id'].split(':')
                        source=sources[origin][int(index)]
                        answer=norm(source.get('answer',''))
                        if not answer: raise ValueError('Missing source answer')
                        keys.extend([('article',origin+':'+index),('answer',sha(answer.encode()))])
            for key in keys:
                if key in owners: union(gid,owners[key])
                else: owners[key]=gid
    for edge in read(V2/'similarity_audit.jsonl'):
        if edge['review']['approved_equivalence']:
            union(edge['pair']['group_a'],edge['pair']['group_b'])
    protected={find(q['group_id']) for s in ('validation','test') for q in questions[s]}
    removed={q['group_id'] for q in questions['train'] if find(q['group_id']) in protected}
    V3.mkdir(exist_ok=True)
    for split in ('validation','test'):
        for name in ('questions.jsonl','grades.jsonl','unjudged_candidates.jsonl'):
            dest=V3/split/name; dest.parent.mkdir(exist_ok=True)
            dest.write_bytes((V2/split/name).read_bytes())
    for name in ('questions.jsonl','grades.jsonl','unjudged_candidates.jsonl'):
        write(V3/'train'/name,[r for r in read(V2/'train'/name) if r['group_id'] not in removed])
    write(V3/'removed_training_groups.jsonl',[{'group_id':g,'reason':'source_or_question_family_touches_validation_or_test'} for g in sorted(removed)])
    files={str(p.relative_to(V3)).replace('\\','/'):{'sha256':sha(p.read_bytes()),'records':len(read(p))} for p in sorted(V3.rglob('*.jsonl'))}
    manifest={'dataset_version':'3.0.0','schema_version':2,'parent_dataset_version':'2.0.0','parent_manifest_sha256':sha((V2/'manifest.json').read_bytes()),'generator_sha256':sha(Path(__file__).read_bytes()),'policy':'remove whole train families touching unchanged validation/test; no reassignment or relabeling','removed_train_groups':len(removed),'train_groups':len(questions['train'])-len(removed),'test_set_blind':False,'validation_test_source_overlap_remaining':True,'limitations':['Public test is a historical regression benchmark, not blind.','Validation/test source overlap is retained to preserve both splits.','Semantic shared-intent review and historical checkpoint provenance remain outstanding.'],'outputs':files}
    (V3/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:manifest[k] for k in ('dataset_version','removed_train_groups','train_groups')},indent=2))

if __name__=='__main__': build()
