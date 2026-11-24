# Quran search dataset splits

This folder contains the current **provisional** training, validation and test sets for evaluating and improving Aayatun’s Search by meaning. Each question includes the Quran verses reviewers found relevant, with their review reasons.

| Split | Question groups | Share | Groups with at least one direct match | Scored question–ayah pairs |
| --- | ---: | ---: | ---: | ---: |
| Training | 1,110 | 64.95% | 137 | 4,104 |
| Validation | 257 | 15.04% | 49 | 980 |
| Test | 342 | 20.01% | 65 | 1,579 |
| **Total** | **1,709** | **100%** | **251** | **6,663** |

A question group holds one question or several versions judged to mean the same thing. A question can have many relevant ayat, so there are more question–ayah pairs than question groups. Groups can have partial or background matches without any direct match.

## Folder guide

```text
split_package/
  train/       questions.jsonl   grades.jsonl
  validation/  questions.jsonl   grades.jsonl
  test/        questions.jsonl   grades.jsonl
  conflicts_excluded_from_grades.jsonl
  create_splits.py
  summary.json
  README.md
```

Each line in `questions.jsonl` contains a group, its question versions, full verse text, original review reasons, source references and potential negatives. Use it to inspect examples and prepare training data. These files preserve all original judgments, **including the 39 disagreements**.

Each line in `grades.jsonl` contains a `group_id`, `ayah` and numeric `grade`. This is the scoring file. The 39 disputed question–ayah grades are excluded from it. The complete questions file still shows the original conflicting grades and `has_conflicting_grades` flag, so **do not use those disputed grades as settled labels for training**.

| Grade | Review meaning |
| --- | --- |
| 3 | Directly addresses the question |
| 2 | Answers part of it or gives a relevant principle |
| 1 | Provides background |
| 0 | Reviewed as unrelated |

For example, “Who goes to heaven?” has four direct ayat in the source dataset, including 40:40 and 19:60. The first promises Paradise to believing men and women who do righteousness. The second says those who repent, believe and do righteousness will enter Paradise. Each match has its own stored reason.

For a full readable example of questions, verse text and reasons, see [the grouped dataset guide](../GROUPED_DATASET.md).

## How the split was made

The target is **65% training, 15% validation and 20% test**, measured by question group. The goal is to use training questions to develop search improvements, validation questions to choose settings, and test questions for the final comparison.

All versions within a question group stay together. To reduce leakage further, question pairs with multilingual E5-small cosine similarity above 0.90 were connected. All questions connected by these links, including through chains, were assigned to the same split. The components were shuffled with seed **4213**, sorted by size and assigned toward the target proportions. `create_splits.py` reproduces the files and `summary.json` records input hashes.

This rule keeps all recorded above-threshold links inside a split. Similar questions below 0.90 can still cross splits, and a similarity link can connect questions that are only broadly related. The largest connected component contains **673 groups**. The current split is therefore a practical starting point, not proof of complete topic separation.

The test questions have already been seen during the model review, so this is a held-out set rather than an independently blind test. Questions and verse labels came from the IslamQA-derived collection and model review. A human subject-matter review, source reuse rights, and an independent review of near-duplicates remain pending.

## Negatives and conflicts

Reviewed unrelated verses are labeled grade 0. Some questions also contain randomly selected candidate negatives. Those candidates are **unreviewed** and could be relevant. They are included for inspection in the questions files, carry a null grade, and are absent from the scoring files.

[conflicts_excluded_from_grades.jsonl](conflicts_excluded_from_grades.jsonl) lists the 39 cases where reviews disagree. Resolve them before restoring their grades. Absence from a scoring file means unjudged for this evaluation, not irrelevant. The dataset is not an exhaustive list of every relevant ayah.

## Compare search versions

Use training data to build an improvement and validation data to select settings. Freeze the chosen approach before testing it on the test set. Compare the existing browser search and the candidate on the same questions and verse corpus. Report the number of questions evaluated and per-question wins and losses.

Measure how many directly relevant ayat appear in the first five and ten results, along with a graded ranking measure that gives direct matches more weight than partial and background matches. Some questions have more than five direct ayat, so a top-five list cannot contain every direct answer. Treat unjudged verses separately when interpreting scores.

## Rebuild

From the repository root run `python datasets/sanitized_islamqa/split_package/create_splits.py`. It reads the grouped data and saved similarity candidates in the parent folder, rebuilds all three splits and checks that each group and every connected component stays in a single split. `summary.json` contains SHA-256 hashes of the inputs. Run it with the same Python version when exact byte-for-byte reproduction is required.
