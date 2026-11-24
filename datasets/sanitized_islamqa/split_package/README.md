# IslamQA Quran search split package v2

This package contains provisional question–ayah relevance data for training and evaluating Aayatun's Quran search. It is designed to be readable, strictly validated, and reproducible.

> **Not release-grade.** The labels are model-assisted, citation corrections are not independently verified, the test questions were seen during review, pooled judgments are incomplete, compound queries were not atomically reviewed, the evaluation scope is English-only, and source reuse rights remain unresolved. See [Release blockers](#release-blockers).

## Current package

| Split | Question groups | Query wordings | Scored pairs | Direct groups | Positive groups |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 1,111 | 1,124 | 4,192 | 161 | 1,093 |
| Validation | 257 | 260 | 1,082 | 39 | 257 |
| Test | 341 | 345 | 1,389 | 51 | 341 |
| **Total** | **1,709** | **1,729** | **6,663** | **251** | **1,691** |

The package also contains 6,702 judged question–ayah pairs, including 39 disputed pairs excluded from authoritative qrels, and 5,043 unjudged candidates stored separately. There are 20 approved equivalence edges. All currently fall inside existing question groups, so the package has 1,709 components. The similarity audit records 820 rejected near-duplicate candidates that cross splits; they are visible for review but do not constrain allocation.

## Folder guide

```text
split_package/
  train/
    questions.jsonl
    grades.jsonl
    unjudged_candidates.jsonl
  validation/
    questions.jsonl
    grades.jsonl
    unjudged_candidates.jsonl
  test/
    questions.jsonl
    grades.jsonl
    unjudged_candidates.jsonl
  conflicts_excluded_from_grades.jsonl
  split_components.jsonl
  similarity_audit.jsonl
  exclusions.jsonl
  manifest.json
  schema.json
  .gitattributes
  dataset_common.py
  create_splits.py
  validate_dataset.py
  load_dataset.py
  evaluate_search.py
  build_review_pool.py
  test_split_package.py
  README.md
```

Each JSONL file contains one complete record per line. JSONL remains compact for tooling; the schema and the example below show the same structure in indented form.

## Schema design

The v2 schema follows four rules:

1. **Identity first.** Stable IDs appear before descriptive content.
2. **Related fields are nested.** Provenance, query statistics, evidence status, judgments, and labels have clear sections.
3. **Constants live once.** Package-wide values such as the schema version live in `manifest.json`. A record's split comes from its directory. Tier order comes from the schema and this README, not a repeated `relevance_order` field.
4. **Derived values are not duplicated.** For example, `label_summary.best_settled_label` replaces separate positive/direct booleans, and component membership replaces a repeated component count.

The strict Draft 2020-12 schema is [schema.json](schema.json). It disallows unknown properties and distinguishes settled, disputed, and unjudged data.

### Question records

`<split>/questions.jsonl` contains one question group per line:

| Field | Meaning |
| --- | --- |
| `group_id` | Stable 16-character hexadecimal question-group ID. |
| `component_id` | Split component that owns the group. |
| `provenance.origin_splits` | Original source splits: `train`, `benchmark`, or both. |
| `provenance.missing_reference_records` | Source IDs whose verified reference is missing; also listed in `exclusions.jsonl`. |
| `queries` | Deduplicated normalized wordings with stable query IDs, language, source IDs, and statistics. |
| `evidence` | Ayat grouped into `direct`, `partial`, `background`, and `unrelated` tiers. |
| `label_summary.best_settled_label` | Highest settled tier in the group, or `none`. |

A query has this shape:

```json
{
  "query_id": "0123456789abcdef:q0",
  "text": "Who will enter Paradise?",
  "language": "en",
  "source_ids": ["train:42:0"],
  "stats": {
    "char_length": 24,
    "question_mark_count": 1,
    "compound_status": "not_flagged"
  }
}
```

`compound_status` is `review_required` when the wording has more than one question mark or exceeds 1,000 characters. This is a review flag, not an automatic split into atomic questions.

Evidence is organized by tier:

```json
{
  "evidence": {
    "direct": [],
    "partial": [],
    "background": [],
    "unrelated": []
  }
}
```

The surrounding tier is the evidence label; it is not repeated inside each verse. A settled verse has an integer `grade` and `status.label_status: "settled_model_review"`. A disputed verse has `grade: null`, `status.label_status: "disputed"`, and a non-authoritative `status.provisional_grade`.

Each judgment preserves its source ID, query ID, original query wording, source reference, reviewed passage, assessment, and reason. This retains the context needed to inspect or correct a label.

### Authoritative qrels

`<split>/grades.jsonl` is the scoring file. It contains settled labels only:

```json
{
  "group_id": "0123456789abcdef",
  "ayah": "1:1",
  "label": "direct",
  "grade": 3
}
```

The join key is `(group_id, ayah)`. Disputed evidence is absent from qrels even though it remains in the question record and conflict file.

| Grade | Label | Meaning |
| ---: | --- | --- |
| 3 | `direct` | Directly addresses the question. |
| 2 | `partial` | Answers part of the question or supplies a relevant principle. |
| 1 | `background` | Provides topical context without answering the question. |
| 0 | `unrelated` | Reviewed as unrelated. |

These are ordinal relevance labels, not probabilities. Verses within a tier are not ranked against one another.

### Disputed and unjudged data

[conflicts_excluded_from_grades.jsonl](conflicts_excluded_from_grades.jsonl) lists all 39 disputed pairs with their competing labels and provisional maximum grade. A conflict is not a settled training label.

`<split>/unjudged_candidates.jsonl` contains random candidates that were not reviewed:

```json
{
  "group_id": "0123456789abcdef",
  "ayah": "2:1",
  "text": "Alif, Lam, Mim."
}
```

The filename and schema define these records as unjudged, so constant `status` and `grade: null` fields are omitted. These candidates never appear in qrels and never overlap judged evidence. Absence from qrels means **unjudged for this evaluation**, not irrelevant.

[exclusions.jsonl](exclusions.jsonl) records 24 unresolved citations or missing verified references, including the original question, source details, reasons, and assigned split when known.

### Components and similarity audit

[split_components.jsonl](split_components.jsonl) maps each component to its split and member group IDs:

```json
{
  "component_id": "cmp_0123456789abcdef",
  "split": "train",
  "group_ids": ["0123456789abcdef"],
  "reason": "balanced_stratified_allocation"
}
```

The alternative reason is `contains_group_without_settled_positive`; those components are always assigned to training.

[similarity_audit.jsonl](similarity_audit.jsonl) groups each candidate into `pair`, `similarity`, `review`, and `split_check` sections. Only `review.approved_equivalence: true` edges constrain splits. Rejected candidates remain diagnostic data, including when they cross splits.

### Complete valid example

The following indented question record and qrel are schema-valid. The text is illustrative; generated package records are checked against the translation corpus.

```json
{
  "group_id": "0123456789abcdef",
  "component_id": "cmp_0123456789abcdef",
  "provenance": {
    "origin_splits": ["train"],
    "missing_reference_records": []
  },
  "queries": [
    {
      "query_id": "0123456789abcdef:q0",
      "text": "Who will enter Paradise?",
      "language": "en",
      "source_ids": ["train:42:0"],
      "stats": {
        "char_length": 24,
        "question_mark_count": 1,
        "compound_status": "not_flagged"
      }
    }
  ],
  "evidence": {
    "direct": [
      {
        "ayah": "1:1",
        "text": "Illustrative verse text.",
        "grade": 3,
        "status": {
          "label_status": "settled_model_review"
        },
        "judgments": [
          {
            "source_id": "train:42:0",
            "query_id": "0123456789abcdef:q0",
            "query": "Who will enter Paradise?",
            "source_reference": "1:1",
            "reviewed_passage": ["1:1"],
            "assessment": {
              "label": "direct",
              "grade": 3,
              "reason": "Illustrative direct relevance judgment."
            }
          }
        ]
      }
    ],
    "partial": [],
    "background": [],
    "unrelated": []
  },
  "label_summary": {
    "best_settled_label": "direct"
  }
}
```

Matching qrel:

```json
{
  "group_id": "0123456789abcdef",
  "ayah": "1:1",
  "label": "direct",
  "grade": 3
}
```

## Split policy

The target is **65% training, 15% validation, and 20% test**, measured by question group.

- Existing question groups are indivisible.
- Exact normalized duplicate questions are kept together.
- Only approved review decisions create equivalence edges between groups.
- Rejected embedding-similarity candidates do not constrain allocation.
- Components containing a group without a settled positive are assigned to training.
- Remaining components are allocated largest-first with seed `4213`, balancing origin profile and best settled label using the largest normalized remaining deficit.

The multilingual E5-small candidate model and revision are pinned in `manifest.json`. The audit file makes both approved and rejected near-duplicates inspectable without reconstructing the graph.

The 18 training groups without a settled positive remain available for inspection but cannot provide a confirmed positive training example until reviewed. Every validation and test group has at least one settled positive.

The test split is **held out but not blind**: its questions were visible during model-assisted review. Freeze a method on validation before examining test results.

## Manifest and reproducibility

[manifest.json](manifest.json) records:

- dataset and schema versions;
- provisional release status;
- generator hash, Python version, source commit, and generation time;
- semantic hashes for every source input;
- output hashes, byte sizes, and record counts;
- split, grade, origin, label, and component-size statistics;
- balanced class weights;
- pinned similarity model and revision;
- translation corpus hash and verse count;
- split policy and allocation seed;
- annotation status and release blockers.

Source JSON/JSONL hashes are computed from parsed canonical JSON, so whitespace and CRLF/LF differences do not change them. Generator hashes use UTF-8 text with normalized LF line endings, so Windows CRLF checkouts validate identically. Output hashes use raw UTF-8 bytes generated with LF line endings, and `.gitattributes` requests LF for JSON, JSONL, Python, and Markdown files. Regenerate with the recorded Python version when exact output hashes are required; `generated_at` is expected to change on every rebuild.

## Validation and loading

Run from the repository root. Replace `<en_quran.db>` with the Aayatun SQLite corpus containing `verses(sura, ayah, text)`.

```powershell
python datasets/sanitized_islamqa/split_package/create_splits.py --corpus "<en_quran.db>"
python datasets/sanitized_islamqa/split_package/validate_dataset.py --check --corpus "<en_quran.db>"
python datasets/sanitized_islamqa/split_package/load_dataset.py
python -m unittest discover -s datasets/sanitized_islamqa/split_package -p "test_*.py" -v
```

The validator checks schema conformance, unique IDs, split ownership, component integrity, query statistics, judgment/query consistency, reviewed passages, grade/tier consistency, disputed null grades, exact qrel/evidence equality, conflict coverage, candidate separation, similarity audits, exclusions, manifest counts/statistics/hashes, and optional corpus text equality.

The loader exposes validated questions, numeric qrels, candidates, conflicts, components, split ownership, and judged-pair sets:

```python
from load_dataset import Dataset

dataset = Dataset.load()
validation = dataset.splits["validation"]
question = validation.questions["0123456789abcdef"]
grades = validation.qrels["0123456789abcdef"]
judged = dataset.judged_pairs_for("0123456789abcdef")
```

## Ranking evaluation

A ranking file is JSONL with one record per evaluated group:

```json
{
  "group_id": "0123456789abcdef",
  "ranking": ["1:1", "2:255", "3:104"]
}
```

Ayah references must be unique within a ranking. One file represents one named run.

```powershell
python datasets/sanitized_islamqa/split_package/evaluate_search.py `
  --split validation `
  --run baseline=baseline_rankings.jsonl `
  --run candidate=candidate_rankings.jsonl `
  --compare baseline candidate `
  --paired-metric ndcg@10 `
  --bootstrap 2000 `
  --output evaluation_results.json
```

Evaluation includes only groups with at least one settled positive. Missing rankings are treated as empty result lists and reported explicitly. Unjudged and disputed results occupy rank positions but earn no gain.

| Metric | Definition |
| --- | --- |
| `direct_hit@5` | At least one grade-3 ayah in the first five results. |
| `useful_hit@5` | At least one grade-2-or-3 ayah in the first five results. |
| `positive_hit@5` | At least one grade-1-or-higher ayah in the first five results. |
| `recall@5`, `recall@10` | Retrieved settled positives divided by all settled positives for the group. |
| `mrr@10` | Reciprocal rank of the first settled positive within ten results. |
| `ndcg@10` | DCG with gain `2^grade - 1`, divided by the ideal DCG from settled qrels. |

Reports contain macro averages, 95% percentile bootstrap confidence intervals from group resampling, per-stratum results, ranked/missing group counts, and result coverage split into settled-judged, disputed, and unjudged results. The default bootstrap uses 2,000 samples and seed `4213`.

Paired comparison reports per-group wins, losses, ties, mean delta, and a paired bootstrap confidence interval for the selected metric. A tie uses a `1e-12` tolerance.

## Human review pool

Merge top-k results from multiple runs into a deduplicated queue:

```powershell
python datasets/sanitized_islamqa/split_package/build_review_pool.py `
  --split validation `
  --run baseline=baseline_rankings.jsonl `
  --run candidate=candidate_rankings.jsonl `
  --top-k 10 `
  --corpus "<en_quran.db>" `
  --output review_pool.jsonl
```

Each queue record contains the split, group ID, all query IDs and texts, ayah, corpus text, and every contributing `(run, rank)` source. Already judged pairs are excluded, including disputed pairs. Duplicate candidates across runs become one record with multiple sources.

Human adjudication must be recorded and applied before pooled judgments can be treated as settled qrels. Building a queue does not make the benchmark release-grade.

## Release blockers

- Independent human or scholarly relevance review is pending.
- Citation corrections are not independently verified.
- Test questions were seen during model review.
- Pooled relevance judgments are incomplete.
- Source reuse rights and licensing are unresolved.
- Compound queries were not atomically reviewed.
- Evaluation scope is English-only.
