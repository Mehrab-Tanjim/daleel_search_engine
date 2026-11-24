# User-held private final evaluation

**Custodian and final evaluator: the user.** Development uses the repaired public train/validation data and the historical public regression set. This is a separate final test, not a renamed subset of the already exposed benchmark.

## Collect and seal privately

1. Collect fresh, independently authored realistic Quran-search questions, or previously unused sources whose exposure can be documented. Do not generate them from this chat, the public test, test failure lists, or paraphrases of existing train/validation examples. Public web questions may still have unknown pretrained-model exposure; do not claim guaranteed absence from model pretraining.
2. In the private environment, compare each candidate against **all** public development material, the legacy train/benchmark inputs, prior public tests, generated questions and known training manifests. Check source article IDs, full-answer fingerprints, exact/near questions and shared atomic intents. Keep entire source/intent families together. Similarity thresholds only nominate pairs for review; do not automatically equate related topics. The same Quran ayah may legitimately answer questions in several sets.
3. Have qualified independent reviewers assign direct/partial/background/unrelated relevance to the frozen 6,236-ayah corpus. Adjudicate disagreements, review compound requests, preserve source evidence, and pool candidates from fixed baseline systems without revealing system identities to reviewers. Keep unknown relevance distinct from unrelated. Do not derive labels from the candidate under development alone.
4. Keep questions, qrels, source answers and review materials outside this repository, this chat and developer-accessible storage. No uploads here are required. Restrict access to the user and appointed independent reviewers. Development may receive the corpus version, schema, metric definitions and predeclared sample sizes, but no private examples.
5. Seal the private files, grouping policy, rubric, scorer version, metric definitions and sample size before final evaluation. Store a timestamped SHA-256 manifest locally. A hash is a tamper check, not access control or proof of blindness. Record who has seen what and any exceptions.

## Freeze the candidate before unsealing

The development handoff must include a commit, model/index/configuration hashes, exact corpus hash, dependency lock, offline inference entry point, rollback assets and public validation/regression results. Fix acceptance criteria and candidate submission count **before** the private evaluation. Do not use private results to choose between a menu of candidates.

Use a local, offline run with no telemetry or network access, no private-query logging to shared files, no training, and no model/index writes. Run the fixed historical baseline and one frozen candidate against identical queries. The inference process receives question text but no labels, source answers or evaluator files. Keep labels in a separate process/account/container; a command-line convention alone is not an access boundary. The user operates the browser or runner and retains private logs. Exact runtime isolation must be verified when the final inference artifact exists.

## Score locally, disclose only the agreed aggregate

- Match every query ID exactly once; fail on missing/extra queries, invalid ayah IDs, duplicate result IDs or mutated artifacts. Preserve raw ranks and map auxiliary representations to unique canonical ayat.
- Reuse the public `score_group` metric definitions from the pinned evaluator, with the private set's own denominators. Report relevant hit@5, useful hit@5 on answerable queries, useful hit@5 on the direct-answer slice, nDCG@10, judgment coverage and paired confidence intervals resampled by private family/group. Predeclare how multiple wordings per family are weighted.
- Share only the agreed aggregate report, counts, provenance attestations and candidate/scorer hashes. Keep private questions, qrels, rankings and failure examples with the user. Label an exposed or repeated adaptive evaluation as such.
- If a private result triggers model/prompt/corpus tuning, that set has become development feedback. Retire it for subsequent blind claims and collect a fresh holdout. Repeating an unchanged artifact solely to diagnose nondeterminism must be logged and must not become a selection loop.

## Ready checklist

- [ ] Private questions and judgments exist and have independent review.
- [ ] Source/intent-family checks against all development material are complete.
- [ ] Custody, access log and sealed manifest exist.
- [ ] Candidate, baseline, corpus, scorer and thresholds are frozen.
- [ ] Offline execution and label/inference separation are verified.
- [ ] The user performs the final run and releases the agreed aggregate only.

Until these are complete, the accurate status is **private final holdout pending**, not “blind test verified.”
