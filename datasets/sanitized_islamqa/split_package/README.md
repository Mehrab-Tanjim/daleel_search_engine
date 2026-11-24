# Quran search dataset splits

This folder contains the current **provisional** training, validation and test sets for evaluating and improving Aayatun’s Search by meaning. Each question includes the Quran verses reviewers found relevant, with their review reasons.

| Split | Question groups | Share | Groups with at least one direct match | Scored question–ayah pairs |
| --- | ---: | ---: | ---: | ---: |
| Training | 1,110 | 64.95% | 138 | 4,115 |
| Validation | 257 | 15.04% | 50 | 1,052 |
| Test | 342 | 20.01% | 63 | 1,496 |
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

Each line in `questions.jsonl` contains a group, its question versions, full verse text, judges' reasons and potential negatives. Each question appears once in the group's `queries` list. Each verse has a `judge_rationales` list containing the review reasons. Source references, the question associated with each review, and the original grades are available in [the grouped audit data](../grouped_queries.jsonl), using the group ID and ayah. The split records flag the 39 disagreements.

For example, a record has this shape (the verse text is shortened here):

```json
{
  "queries": ["Who goes to heaven?"],
  "ayat": {
    "direct": [{
      "ayah": "40:40",
      "text": "Whoever does righteousness ... will enter Paradise ...",
      "grade": 3,
      "judge_rationales": [{
        "reason": "Believing men and women who act righteously are explicitly promised Paradise."
      }]
    }]
  }
}
```

The full record includes all tiers, other matching verses and every review reason. A verse can have multiple rationales when it was cited or reviewed more than once. The grouped audit data retains the source ID, original reviewed passage and associated query for each reason.


Each line in `grades.jsonl` contains a `group_id`, `ayah` and numeric `grade`. This is the scoring file. The 39 disputed question–ayah grades are excluded from it. The complete questions file still shows the original conflicting grades and `has_conflicting_grades` flag, so **do not use those disputed grades as settled labels for training**.

| Grade | Review meaning |
| --- | --- |
| 3 | Directly addresses the question |
| 2 | Answers part of it or gives a relevant principle |
| 1 | Provides background |
| 0 | Reviewed as unrelated |

For example, “Who goes to heaven?” has four direct ayat in the source dataset, including 40:40 and 19:60. The first promises Paradise to believing men and women who do righteousness. The second says those who repent, believe and do righteousness will enter Paradise. Each match has its own stored reason.

For a full readable example of questions, verse text and reasons, see [the grouped dataset guide](../GROUPED_DATASET.md).

## How the questions were collected

The original Daleel collection used IslamQA question-and-answer material from two sources. The reference-extraction script loads the Hugging Face `minhalvp/islamqa` training data and local Kaggle IslamQA files. It removes Kaggle questions that match a training question after trimming and lowercasing. It writes separate original training and benchmark reference files. The present splits were made later from their reviewed contents. [Reference-extraction script](../../extract_daleel_from_islamqa.py)

The repository also contains an earlier IslamQA scraping script. It requests topic pages, follows pagination, collects question IDs and attempts to fetch answer pages. Its intended output includes the page ID, URL, title, question, answer summary and full answer. The checked-in script builds answer URLs from the loop index rather than the collected question ID. Its comments and output filename also do not establish which topic range was actually captured. These details limit what can be concluded about the completeness and provenance of the saved scrape. We have not established that every item in the two reference files came from that script. [Scraping script](../../scrap_islamqa_dataset.py)

The extraction script sends each question and answer to `gpt-4o`, asking it to extract quoted Quran verses and hadith references. It saves the question, answer and extracted citations. A citation shows what the answer cited. It does not by itself establish that the verse answers the reader's particular question. The script can also return an empty citation list on an error or when its expected JSON format is absent. The saved files are therefore not a complete inventory of every verse that may be relevant.

## How we reviewed and organized the pairs

We started with the two saved reference files and selected the **5,343 citations labeled Quran**. Reviewers compared each question and citation with the English verse text in the Aayatun search database and recorded a decision and reason. Where the cited number was missing or contradicted the quoted text, reviewers recorded corrections when they could identify the reference. The original reference remains in the data. For example, one citation labeled 23:40 quoted a passage from 40:40. Citation corrections still need independent verification.

| First review decision | Citations | Why it was assigned |
| --- | ---: | --- |
| Direct | 761 | The cited verse or passage addresses the question or its main topic. |
| Partial | 3,379 | It addresses part of the question or offers a relevant principle, while the specific answer needs more context. |
| Background | 1,151 | It relates to the topic but does not answer the question. |
| Unrelated | 29 | The reviewer found no useful connection to the question. |
| Unresolved | 23 | The question or reference could not be judged reliably. |

These are initial citation counts, before expanding ranges into individual ayat or combining question variants. The 23 unresolved citations were omitted from the grouped dataset. The full review decisions are saved in [the audit file](../all_pairs_reviewed.jsonl).

We kept all different question wordings. Exact normalized matches were grouped, and multilingual E5-small was used to find question pairs with cosine similarity above 0.90. Model reviewers then checked those candidate pairs for equivalent intent. Similarity alone did not cause a merge. Every pair across a merged group required approval, avoiding automatic merges through a chain of merely similar questions. The [grouped dataset guide](../GROUPED_DATASET.md) describes the saved decisions and conservative overrides.

For each group, the data keeps its direct, partial, background and unrelated ayat together with the original review reasons. Verse ranges are expanded into individual ayat for ranking, while `reviewed_passage` retains the original range. A grade assigned to a passage can depend on its surrounding ayat, so interpret an individual verse alongside that retained context.

## Why the grades have this order

The ranking reflects how useful a verse is for answering the reader's question. A **direct** answer should rank ahead of a verse that answers only part of the question. A **partial** answer should rank ahead of background that merely concerns the same topic. An explicitly reviewed **unrelated** verse receives grade 0 and can serve as a negative example.

For example, 40:40 was graded direct for “Who goes to heaven?” because it explicitly promises Paradise to believing men and women who do righteous deeds. For a question asking the travel distance required to shorten prayer, 4:101 was graded partial. It addresses shortening prayer during travel, but does not state a minimum distance. For a question asking the Ramadan holiday date, 2:189 was graded background. It discusses the new moons as markers of time, but does not give the requested date.

The numbers **3, 2, 1 and 0** encode that order for training and evaluation. They are ordinal labels, not measured probabilities or an assertion that one verse is exactly three times as useful as another. Verses within the same grade were not ordered against each other. Reviewers can disagree, especially when a question asks for a legal ruling and a verse supplies a general principle. The 39 disputed grades are excluded from the scoring files until resolved.

The initial reviews were model-assisted. Another model reviewed the first 53 pairs and direct candidates in two review groups, and several decisions were changed after that check. This is not an independent human or scholarly review. The dataset retains the reasons and original evidence so a reader can inspect or correct the decisions.

## How the split was made

The target is **65% training, 15% validation and 20% test**, measured by question group. The goal is to use training questions to develop search improvements, validation questions to choose settings, and test questions for the final comparison.

All versions within a question group stay together. To reduce leakage further, question pairs with multilingual E5-small cosine similarity above 0.90 were connected. All questions connected by these links, including through chains, were assigned to the same split. Components containing any question without an undisputed positive grade are assigned to training. This ensures that every validation and test question has at least one direct, partial or background grade in its scoring file. The other components were shuffled with seed **4213**, sorted by size and assigned toward the target proportions. `create_splits.py` reproduces the files and `summary.json` records input hashes.

This rule keeps all recorded above-threshold links inside a split. Similar questions below 0.90 can still cross splits, and a similarity link can connect questions that are only broadly related. The largest connected component contains **673 groups**. The current split is therefore a practical starting point, not proof of complete topic separation.

The training split contains 18 groups without an undisputed positive scoring grade. Those groups cannot supply a confirmed positive for a training example unless their labels are reviewed. They remain in training to keep connected question groups together.

The test questions have already been seen during the model review, so this is a held-out set rather than an independently blind test. Questions and verse labels came from the IslamQA-derived collection and model review. A human subject-matter review, source reuse rights, and an independent review of near-duplicates remain pending.

## Negatives and conflicts

Reviewed unrelated verses are labeled grade 0. Some questions also contain randomly selected candidate negatives. Those candidates are **unreviewed** and could be relevant. They are included for inspection in the questions files, carry a null grade, and are absent from the scoring files.

[conflicts_excluded_from_grades.jsonl](conflicts_excluded_from_grades.jsonl) lists the 39 cases where reviews disagree. Resolve them before restoring their grades. Absence from a scoring file means unjudged for this evaluation, not irrelevant. The dataset is not an exhaustive list of every relevant ayah.

## Compare search versions

Use training data to build an improvement and validation data to select settings. Freeze the chosen approach before testing it on the test set. Compare the existing browser search and the candidate on the same questions and verse corpus. Report the number of questions evaluated and per-question wins and losses.

Measure how many directly relevant ayat appear in the first five and ten results, along with a graded ranking measure that gives direct matches more weight than partial and background matches. Some questions have more than five direct ayat, so a top-five list cannot contain every direct answer. Treat unjudged verses separately when interpreting scores.

## Rebuild

From the repository root run `python datasets/sanitized_islamqa/split_package/create_splits.py`. It reads the grouped data and saved similarity candidates in the parent folder, rebuilds all three splits and checks that each group and every connected component stays in a single split. `summary.json` contains SHA-256 hashes of the inputs. Run it with the same Python version when exact byte-for-byte reproduction is required.
