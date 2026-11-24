# IslamQA split leakage and blind-holdout audit

Audited dataset: `7484cfa2eae1cfa399e1c46ad656396b2c9d4b0e`.

**Verdict: the current splits are not source-family independent, and the current test is not blind.** Passing the existing schema validator did not establish either property. Do not certify the benchmark as leakage-free or use its scores as a blind generalization estimate.

## Split leakage

The existing validator passes against Aayatun's English Quran database, and its 15 tests pass. An independent audit finds zero cross-split group IDs, component IDs, query IDs, normalized exact questions, source-reference IDs or original source-row identities.

However, **29 full source-answer families cross splits, with byte-identical answers under different question wordings/titles**:

| Boundary | Answer families |
| --- | ---: |
| Train / test | 14 |
| Train / validation | 11 |
| Validation / test | 4 |

This affects 58 groups: 25 train, 15 validation and 18 test. These are repeated source answers, not merely shared Quran verses. For example, train group `83f6957c397cbc6c` asks who wrote the Quran and how it was assembled; test group `359fee25a5ee6f99` asks who wrote it. Both have the same complete source answer. Normalized answer SHA-256: `262fe3053ce222edc0eeb31c01ff2221a51d0b3322f8abe496ff6c7029bf74da`.

The grouping policy considered question equivalence but did not enforce shared source-answer/article families. A difference in question scope can justify separate relevance judgments while still requiring the questions to stay in the same split family. **Do not merge their qrels merely to keep their split allocation together.**

The existing semantic audit also contains 820 rejected cross-split similarity candidates (418 train/test, 317 train/validation, 85 validation/test). These are candidates, not 820 proven leaks. A separate character 3–5-gram TF-IDF screen at cosine >= 0.85 found one pair, which changes the subject from men to women. Neither an embedding threshold nor lexical overlap alone proves equivalence. Source-family and shared atomic-intent review remains necessary, especially for compound questions.

## Historical training exposure

189/257 current validation groups and 252/341 current test groups match questions in `datasets/islamqa_references_train.json`, the legacy trainer's default input. The v2 split was created after those original train/benchmark inputs existed.

`finetune.py` expands questions into multiple positive-reference pairs, then performs a random pair-level 80/20 split. Reconstructing its pair builder using the Hugging Face datasets 3.6.0 non-stratified split algorithm and seed 42 gives:

- 8,303 pairs: 6,642 train and 1,661 validation.
- 876 normalized questions shared between those two sets.
- 1,498/1,661 validation pairs have a question present in training.
- 242 current test groups and 179 current validation groups occur in the reconstructed legacy training partition.

This demonstrates a leakage-prone training path. It does **not** prove which rows an old checkpoint actually consumed; historical dependency versions and training manifests are missing. Existing `islamqa_fine_tuned_*` indices/checkpoints must not be treated as clean baselines without lineage evidence. PR #2's pretrained E5-small is not shown by this audit to have been fine-tuned on these examples. Pretraining contamination is unknown.

Split algorithm reference: [Hugging Face datasets 3.6.0](https://github.com/huggingface/datasets/blob/3.6.0/src/datasets/arrow_dataset.py#L4364).

## Test blindness

The dataset manifest already declares `annotation.test_set_blind: false`. Questions and qrels are public, the questions were seen during review, and all 345 wordings have saved baseline/candidate rankings in Aayatun's evaluation artifacts. The test has also informed this improvement plan.

Reshuffling, renaming, deleting public files, encrypting a previously seen set or changing the manifest flag cannot undo this exposure. Retain the original test as an explicitly **public historical regression benchmark**. Repeated promotion decisions based on it are development feedback, even when each candidate is frozen before a run.

The user has chosen to hold a **new private final holdout and run the final evaluation**. That holdout has not been collected or evaluated in this work. Its custody protocol is in [PRIVATE_HOLDOUT_PROTOCOL.md](PRIVATE_HOLDOUT_PROTOCOL.md). Development must not read its questions, labels, source answers or per-query results.

## Reproduce the source-family gate

From the repository root, with Python 3.10+:

```text
python -m unittest discover -s evaluation -p "test_audit_islamqa_leakage.py" -v
python evaluation/audit_islamqa_leakage.py --output evaluation/islamqa_leakage_audit.json
python evaluation/audit_islamqa_leakage.py --output evaluation/islamqa_leakage_audit.json --require-blind
```

The audit intentionally exits **1** on the current package. `--require-blind` also fails because private custody cannot be certified from these files, even if someone flips the manifest flag. The script uses the Python standard library and saves IDs/hashes rather than question/answer text. The checked-in JSON contains every duplicate family and input hashes. Run the existing package validator separately; the new check supplements it.

## Required repair before training or promotion

1. Preserve the published v2 package and historical leaderboard. Create a versioned development allocation constrained by source article identity, exact full-answer fingerprints, approved paraphrases and reviewed shared atomic intents. Keep relevance groups separate inside larger split families.
2. Restore source URL/article IDs where available; hashes are a useful guard, not complete source lineage. Independently review suspected semantic families and compound questions.
3. Replace the legacy trainer's raw input and pair-level split with only the approved train/validation allocation. No test rows, test-derived paraphrases, disputed negatives or unjudged negatives. Freeze corpus-only synthetic representations separately from question-derived training data.
4. Start from a pinned clean pretrained checkpoint unless complete historical training provenance proves compatibility. Keep all input, family, model and generation manifests.
5. Run the new source-family audit, original validator and regression suites. Zero exact overlaps is necessary but does not certify semantic independence. The user's private test remains a separate final evaluation with different denominators.

No split files, qrels, historical scores or trained models were changed by this audit. Source-family repair and private-holdout creation remain outstanding; this report does not claim that leakage has been removed.
