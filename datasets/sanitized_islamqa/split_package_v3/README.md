# IslamQA dataset v3.0.0

This version removes 25 training groups whose source/question families touch validation or test. Training now has 1,086 groups. Validation and test questions, grades and unjudged candidates are copied byte-for-byte from v2. The published v2 package remains unchanged.

Dataset version is 3.0.0; the record schema remains version 2 because record fields have not changed. Stable IDs and labels are preserved. `removed_training_groups.jsonl` records every removed group. The manifest pins the parent manifest, generator and output hashes.

From the repository root:

```text
python datasets/sanitized_islamqa/build_v3.py
python datasets/sanitized_islamqa/validate_v3.py
```

Consume the JSONL files under this directory directly, joining grades by `(group_id, ayah)`. The v2 `Dataset.load()` default still refers to the historical v2 package; do not use it implicitly for v3 training. The v2 manifest validator is not a v3 validator: use `validate_v3.py`.

Sanitization joins whole families through identical normalized full source answers, original source article rows, normalized questions, existing components and approved equivalence edges. It removes all training members of any family touching either protected split. It never moves validation/test examples into training or changes judgments.

Four identical-answer families still connect validation and test; these are intentionally retained under the requested preservation policy. Semantic shared-intent review and historical checkpoint lineage remain unresolved. This package does not certify universal absence of leakage or pretrained-model contamination. The public test is already exposed and remains a historical regression benchmark. The user's private final holdout is separate and pending.
