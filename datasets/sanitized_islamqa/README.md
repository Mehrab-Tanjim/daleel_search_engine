# IslamQA verse review

All 5,343 extracted Quran citations were reviewed once by model reviewers. The first 53 pairs and direct decisions in two review groups received an additional model review. This is not a human or scholarly review. The files retain each source citation, the verse text, the decision and the reason.

| Decision | Pairs |
| --- | ---: |
| Directly addresses the question | 761 |
| Addresses part of the question | 3,379 |
| Background | 1,151 |
| Unrelated | 29 |
| Unable to resolve | 23 |

The direct files contain 734 pairs after removing 27 duplicates. A question can have more than one verse or range. Counts below are unique question-and-reference pairs and distinct questions.

| File | Pairs | Questions |
| --- | ---: | ---: |
| train_direct.jsonl | 458 | 156 |
| validation_direct.jsonl | 107 | 33 |
| benchmark_direct.jsonl | 169 | 80 |

The original benchmark stays separate. The original training questions were assigned by a deterministic hash of their normalized text to training or validation, approximately 85/15 by question. Identical normalized questions stay together. Similar rewordings and closely related topics have not been checked for overlap. The benchmark questions and answers have already been seen during model review, so this is not a blind independent test. Some questions have many referenced verses, and the counts should not be interpreted as balanced coverage.

all_pairs_reviewed.jsonl contains the complete audit and exclusions. The individual review decisions retain original IDs. The source extraction includes hadith and non-Quran material; this work covers Quran-labeled citations only. Verify rights to reuse IslamQA text and obtain subject-matter review before deploying or treating these labels as authoritative. Reviewers may disagree about whether a verse directly answers a legal question. Citation repairs should also receive verification against an authoritative Quran source.

To reproduce the three files, combine the three decision files into review_decisions.jsonl, then run `python datasets/sanitize_islamqa.py assemble --corpus PATH_TO_SEARCH_DB`. The script requires the Aayatun SQLite search index and refuses to assemble if review coverage is incomplete.

The newer grouped dataset keeps query variants and all graded matches together. See [GROUPED_DATASET.md](GROUPED_DATASET.md) for the format and validation. Earlier direct-only train/validation/benchmark files are historical outputs, and the grouped data needs a new group-level split.
