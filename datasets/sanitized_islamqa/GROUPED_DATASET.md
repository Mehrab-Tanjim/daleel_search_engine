# Grouped questions with graded verse matches

This dataset groups repeated questions and reviewed equivalent phrasings. All original question wordings and all 5,320 source judgments remaining after exclusion of 23 unresolved citations are preserved. One retained source judgment still lacks a usable verse reference and is recorded under missing_reference_records.

## Counts

- 1,709 question groups containing 1,731 wordings.
- 22 groups contain multiple wordings.
- 265 groups have direct matches, 1,373 have partial matches, 664 have background matches and 28 have reviewed unrelated matches. These counts overlap.
- 6,702 distinct group-and-ayah judgments. Verse ranges are expanded to individual ayat with the original reviewed_passage retained as context.
- 39 group-and-ayah grade conflicts, retained with all source reasons. The highest reviewed grade determines the display tier. Review conflicts before using them as training truth.

## Format

Each line in grouped_queries.jsonl contains one group. queries is an array of all its wordings. ayat contains direct, partial, background and unrelated arrays, in that relevance order. Each ayah has text, numeric grade, source judgments and their reasons. Grades are 3, 2, 1 and 0 respectively. Ayat with the same grade have no reviewed order within their tier.

The same verse appears only once in a group's tiers, but all original judgments and source IDs are retained underneath it. If source judgments disagree, has_conflicting_grades is true. grouped_conflicts.jsonl lists these cases.

Reviewed unrelated matches are negative examples. For groups without those matches, random_negative_candidates contains three deterministic random verses, excluding all judged verses and their immediate neighbors. These carry review=random_unjudged and grade=null. Random sampling can select a relevant verse, so these candidates need checking before being treated as confirmed negatives. They are excluded from graded qrels.

## Grouping decisions

Pinned multilingual E5-small query embeddings found 1,755 pairs above cosine 0.90. All candidate pairs received model equivalence decisions, preserved in group_review_0.json through group_review_2.json. Four conservative parent-review overrides are recorded separately. Similar topics with different gender, constraints or subquestions stay separate. Every pair across a proposed merged group must have explicit equivalence approval, preventing transitive similarity chains from merging unrelated questions. Normalization groups case and punctuation variants while preserving both wordings.

The similarity model is intfloat/multilingual-e5-small at c007d7ef6fd86656326059b28395a7a03a7c5846. The script uses query prefixes, normalized mean-pooled vectors and a 512-token cap. group_questions.json preserves the embedding inputs and similar_question_candidates.json preserves the scores.

## Training and evaluation

Use grouped_queries.jsonl for the full ranked lists and reasons. grouped_qrels.jsonl provides group ID, ayah, grade and conflict indicator. source_splits preserves provenance, not a new train/test assignment. Twelve groups span the original source splits, so the earlier direct-only split files should not be used alongside these groups as a leakage-safe partition. Assign complete groups to new splits in the next step. Similar topics below the threshold may still overlap.

Fifty groups contain more than five direct ayat. It is impossible for all their direct ayat to appear in five result slots. Evaluate recall at an appropriate cutoff as well as graded nDCG. For groups with four direct ayat, recall@5 tests whether all four appear.

These are model-reviewed relevance judgments. Human subject-matter review and verification of reuse rights remain pending. The benchmark has been seen during data review, so it is not an independently blind test.

To rebuild, run python datasets/build_grouped_islamqa.py --corpus PATH_TO_AAYATUN_SEARCH_DB. The build refuses to proceed without all similarity decisions. Existing source files and earlier review outputs remain available for audit.
