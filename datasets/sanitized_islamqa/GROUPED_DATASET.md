# Quran search dataset

This dataset connects questions to Quran verses. For each question, it keeps every reviewed verse, the reason it was selected, and how closely it answers the question. It can be used to prepare search training and evaluation data.

## See an example

The question **“Who goes to heaven?”** appears once in the dataset, with four verses marked as direct answers.

| Question | Ayah | English translation | Match | Review reason |
| --- | --- | --- | --- | --- |
| Who goes to heaven? | 40:40 | Whoever does righteousness, whether male or female, while a believer, will enter Paradise. | Direct | Believing men and women who act righteously are promised Paradise. |
| Who goes to heaven? | 19:60 | Those who repent, believe and do righteousness will enter Paradise. | Direct | Repentance, belief and righteous deeds are linked to Paradise. |
| Who goes to heaven? | 46:13 | Those who say “Our Lord is Allah” and remain on a right course will have no fear or grief. | Direct | This verse identifies the people described in the next verse. |
| Who goes to heaven? | 46:14 | Those are the companions of Paradise, abiding there as a reward for their deeds. | Direct | Together with 46:13, it promises Paradise to the steadfast. |

Translations in this table are shortened for readability. The data files include the full English translation, original citation and review reason. Verses 46:13 and 46:14 were reviewed **together as a passage**. The original passage reference is retained for both.

A person can ask the same question in different words. We keep those alternatives together so they can all be used in training. For example, the dataset groups these two questions:

| Question versions kept together | Ayah | English translation | Match | Review reason |
| --- | --- | --- | --- | --- |
| Is smoking cigarettes Haraam? · Why is smoking haram? | 4:29 | Do not consume one another’s wealth unjustly, and do not kill yourselves or one another. | Partial | The prohibition on killing oneself is relevant to harm, but the verse does not name cigarettes. |
| Is smoking cigarettes Haraam? · Why is smoking haram? | 17:26 | Give relatives, the poor and travellers their rights, and do not spend wastefully. | Partial | Wastefulness is relevant to the argument, but tobacco is not named. |

The dataset records which wording each reason came from. Two questions were grouped only after a reviewer judged that they asked the same thing. A similarity score helped find candidates for review.

## What the numbers mean

| Count | Plain-language meaning |
| --- | --- |
| **1,709 question groups** | The dataset has 1,709 sets of questions. Each set represents one question or a few questions judged to mean the same thing. |
| **1,731 question wordings** | Counting every different way the questions were written gives 1,731. The smoking example above contributes one group and two wordings. |
| **22 groups with multiple wordings** | In 22 groups, two or more different wordings were kept together. |
| **6,702 question–ayah matches** | This counts each ayah matched to each question group. “Who goes to heaven?” contributes four matches. A verse range was counted one ayah at a time. |

The groups can contain several types of match at once. That is why the following counts overlap.

| Review | Meaning | Groups containing at least one |
| --- | --- | ---: |
| Direct | The ayah directly addresses the question. | 265 |
| Partial | The ayah answers part of it or gives a relevant principle. | 1,373 |
| Background | The ayah supplies context but does not answer it. | 664 |
| Unrelated | The reviewed ayah does not answer the question. | 28 |

The data gives direct matches grade **3**, partial **2**, background **1**, and unrelated **0**. Verses within the same grade have not been ranked against each other.

## How to read a record

Each line in [grouped_queries.jsonl](grouped_queries.jsonl) is one question group. Its main fields are:

| Field | Contents |
| --- | --- |
| `queries` | Every question wording kept in the group. |
| `ayat` | Four lists named `direct`, `partial`, `background` and `unrelated`. Each verse includes its text and review reasons. |
| `judgments` | The original query, source citation, grade and reason for a verse. |
| `random_negative_candidates` | Random verses supplied when the group has no reviewed unrelated example. |
| `source_splits` | Whether the original questions came from the earlier training or benchmark file. |

Random candidates have **not been checked for relevance**. They might answer the question, so verify them before using them as negatives. The reviewed unrelated examples have an actual review decision.

## Review decisions needing attention

There are **39 question-and-ayah combinations with conflicting grades**. For instance, two source reviews may grade the same verse differently for a grouped question. The dataset currently displays the highest grade and preserves both reasons. See [grouped_conflicts.jsonl](grouped_conflicts.jsonl) before using these cases for training.

All 5,320 reviewed source citations remain traceable in the groups. The 23 citations marked unresolved were excluded. One remaining record lacks a usable verse reference and is listed in its group’s `missing_reference_records`.

The reviews were made by models. Human subject-matter review, source reuse rights, and checks for similar questions across future data splits remain pending.

## Files and future testing

- [Grouped questions, verses and reasons](grouped_queries.jsonl)
- [Compact question–ayah grades](grouped_qrels.jsonl)
- [Conflicting grades](grouped_conflicts.jsonl)
- [Dataset counts](grouped_summary.json)

The earlier training, validation and benchmark files were created before grouping. **Twelve groups contain questions from more than one of those original splits**, so make new splits by whole group before training or comparing models. The current benchmark has been seen during review and is not an independently blind test.

Fifty questions have more than five direct ayat. A search cannot place all of them in five slots. Measure how many relevant ayat appear at several result counts, and use graded ranking measures alongside that count.

To rebuild the grouped files, run `python datasets/build_grouped_islamqa.py --corpus PATH_TO_AAYATUN_SEARCH_DB`. The script uses the saved question-similarity reviews and checks that every candidate has a decision. Similarity candidates were found with the pinned multilingual E5-small model described in [find_similar_questions.py](../find_similar_questions.py).
