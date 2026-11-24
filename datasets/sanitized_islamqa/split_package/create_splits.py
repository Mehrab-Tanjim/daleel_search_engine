"""Build the validated v2 IslamQA question-ayah splits."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import random
import sqlite3
import subprocess
from typing import Any

from dataset_common import (
    DATA,
    GRADES,
    PACKAGE,
    SCHEMA_VERSION,
    SPLITS,
    TIERS,
    ayah_parts,
    canonical_json_bytes,
    detect_language,
    file_sha256,
    normalize_text,
    origin_profile,
    read_json,
    read_jsonl,
    semantic_sha256,
    stable_id,
    write_json,
    write_jsonl,
)

RATIOS = {"train": 0.65, "validation": 0.15, "test": 0.20}
SEED = 4213
SIMILARITY_MODEL = "intfloat/multilingual-e5-small"
SIMILARITY_REVISION = "c007d7ef6fd86656326059b28395a7a03a7c5846"
SIMILARITY_THRESHOLD = 0.90
REVIEW_SHARDS = ("group_review_0.json", "group_review_1.json", "group_review_2.json")
SOURCE_INPUTS = (
    "grouped_queries.jsonl",
    "group_questions.json",
    "similar_question_candidates.json",
    "grouped_conflicts.jsonl",
    "all_pairs_reviewed.jsonl",
    *REVIEW_SHARDS,
    "group_review_overrides.json",
)


def load_review_decisions() -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    candidates = read_json(DATA / "similar_question_candidates.json")
    decisions: dict[int, dict[str, Any]] = {}
    for name in REVIEW_SHARDS:
        for row in read_json(DATA / name):
            index = row["index"]
            if index in decisions:
                raise ValueError(f"Duplicate review decision for candidate {index}")
            if not isinstance(row.get("merge"), bool) or not row.get("reason", "").strip():
                raise ValueError(f"Invalid review decision for candidate {index}")
            decisions[index] = row
    for row in read_json(DATA / "group_review_overrides.json"):
        index = row["index"]
        if not isinstance(row.get("merge"), bool) or not row.get("reason", "").strip():
            raise ValueError(f"Invalid review override for candidate {index}")
        decisions[index] = row
    expected = set(range(len(candidates)))
    if set(decisions) != expected:
        missing = sorted(expected - set(decisions))
        extra = sorted(set(decisions) - expected)
        raise ValueError(f"Review coverage mismatch; missing={missing[:5]}, extra={extra[:5]}")
    return candidates, decisions


def settled_positive(group: dict[str, Any], excluded: set[tuple[str, str]]) -> bool:
    return any(
        (group["id"], verse["ayah"]) not in excluded
        for tier in ("direct", "partial", "background")
        for verse in group["ayat"][tier]
    )


def group_label_stratum(
    group: dict[str, Any], excluded: set[tuple[str, str]]
) -> str:
    for tier in ("direct", "partial", "background"):
        if any(
            (group["id"], verse["ayah"]) not in excluded
            for verse in group["ayat"][tier]
        ):
            return tier
    return "none"


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def root(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        self.parent[self.root(left)] = self.root(right)


def build_components(
    groups: list[dict[str, Any]],
    questions: list[str],
    candidates: list[dict[str, Any]],
    decisions: dict[int, dict[str, Any]],
    excluded: set[tuple[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    normalized_questions = [normalize_text(question) for question in questions]
    if len(set(normalized_questions)) != len(normalized_questions):
        raise ValueError("group_questions.json contains normalized duplicate questions")
    question_index = {question: index for index, question in enumerate(normalized_questions)}

    question_uf = UnionFind(len(questions))
    for index, candidate in enumerate(candidates):
        if decisions[index]["merge"]:
            question_uf.union(candidate["a"], candidate["b"])

    groups_by_id = {group["id"]: group for group in groups}
    if len(groups_by_id) != len(groups):
        raise ValueError("Duplicate group IDs in grouped_queries.jsonl")
    group_ids = list(groups_by_id)
    group_index = {group_id: index for index, group_id in enumerate(group_ids)}
    group_uf = UnionFind(len(groups))
    question_to_group: dict[str, str] = {}

    by_question_root: dict[int, list[str]] = defaultdict(list)
    for group in groups:
        roots = set()
        for query in group["queries"]:
            normalized = normalize_text(query)
            if normalized not in question_index:
                raise ValueError(f"Group query is absent from group_questions.json: {query!r}")
            roots.add(question_uf.root(question_index[normalized]))
            previous = question_to_group.setdefault(normalized, group["id"])
            if previous != group["id"]:
                group_uf.union(group_index[previous], group_index[group["id"]])
        if len(roots) != 1:
            raise ValueError(f"Approved-equivalence graph splits group {group['id']}")
        by_question_root[next(iter(roots))].append(group["id"])

    for linked_groups in by_question_root.values():
        first = linked_groups[0]
        for group_id in linked_groups[1:]:
            group_uf.union(group_index[first], group_index[group_id])

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for group in groups:
        grouped[group_uf.root(group_index[group["id"]])].append(group)

    components = []
    component_by_group: dict[str, str] = {}
    for members in grouped.values():
        members = sorted(members, key=lambda group: group["id"])
        component_id = stable_id("cmp_", *(group["id"] for group in members))
        forced = any(not settled_positive(group, excluded) for group in members)
        reason = (
            "contains_group_without_settled_positive"
            if forced
            else "balanced_stratified_allocation"
        )
        components.append(
            {
                "component_id": component_id,
                "groups": members,
                "forced_to_train": forced,
                "reason": reason,
            }
        )
        for group in members:
            component_by_group[group["id"]] = component_id
    components.sort(key=lambda component: component["component_id"])
    return components, component_by_group


def component_features(component: dict[str, Any], excluded: set[tuple[str, str]]) -> Counter:
    features: Counter = Counter()
    for group in component["groups"]:
        profile = origin_profile(group["source_splits"])
        stratum = group_label_stratum(group, excluded)
        features[(profile, stratum)] += 1
    return features


def allocate_splits(
    components: list[dict[str, Any]], excluded: set[tuple[str, str]]
) -> dict[str, str]:
    total = sum(len(component["groups"]) for component in components)
    targets = {split: total * ratio for split, ratio in RATIOS.items()}
    target_features: Counter = Counter()
    for component in components:
        for feature, count in component_features(component, excluded).items():
            for split, ratio in RATIOS.items():
                target_features[(split, feature)] += count * ratio

    current_total = Counter({split: 0 for split in SPLITS})
    current_features: dict[str, Counter] = {split: Counter() for split in SPLITS}
    ownership: dict[str, str] = {}

    def place(component: dict[str, Any], split: str) -> None:
        current_total[split] += len(component["groups"])
        current_features[split].update(component_features(component, excluded))
        for group in component["groups"]:
            ownership[group["id"]] = split

    for component in components:
        if component["forced_to_train"]:
            place(component, "train")

    remaining = [component for component in components if not component["forced_to_train"]]
    random.Random(SEED).shuffle(remaining)
    remaining.sort(key=lambda component: (-len(component["groups"]), component["component_id"]))

    def deficit(split: str, features: Counter) -> float:
        score = max(0.0, targets[split] - current_total[split]) / targets[split]
        for feature in features:
            target = target_features[(split, feature)]
            if target:
                remaining = max(
                    0.0, target - current_features[split][feature]
                )
                score += remaining / target
        return score

    for component in remaining:
        features = component_features(component, excluded)
        split = max(
            SPLITS,
            key=lambda name: (deficit(name, features), -SPLITS.index(name)),
        )
        place(component, split)

    for group_id, split in ownership.items():
        if split in ("validation", "test"):
            group = next(
                group
                for component in components
                for group in component["groups"]
                if group["id"] == group_id
            )
            if not settled_positive(group, excluded):
                raise ValueError(f"Evaluation group lacks a settled positive: {group_id}")
    return ownership


def dedupe_queries(group: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    source_ids: dict[str, set[str]] = defaultdict(set)
    for verses in group["ayat"].values():
        for verse in verses:
            for judgment in verse["judgments"]:
                source_ids[normalize_text(judgment["query"])].add(judgment["source_id"])

    queries = []
    query_by_normalized: dict[str, dict[str, Any]] = {}
    seen = set()
    for text in group["queries"]:
        normalized = normalize_text(text)
        if normalized in seen:
            continue
        seen.add(normalized)
        question_marks = text.count("?")
        query = {
            "query_id": f"{group['id']}:q{len(queries)}",
            "text": text,
            "language": detect_language(text),
            "source_ids": sorted(source_ids.get(normalized, set())),
            "stats": {
                "char_length": len(text),
                "question_mark_count": question_marks,
                "compound_status": (
                    "review_required"
                    if question_marks > 1 or len(text) > 1000
                    else "not_flagged"
                ),
            },
        }
        queries.append(query)
        query_by_normalized[normalized] = query
    if not queries:
        raise ValueError(f"Group {group['id']} has no queries")
    return queries, query_by_normalized


def format_judgment(
    judgment: dict[str, Any], query_by_normalized: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    query = query_by_normalized.get(normalize_text(judgment["query"]))
    if query is None:
        raise ValueError("Judgment query is absent from its group's deduplicated queries")
    return {
        "source_id": judgment["source_id"],
        "query_id": query["query_id"],
        "query": judgment["query"],
        "source_reference": judgment["source_reference"],
        "reviewed_passage": judgment["reviewed_passage"],
        "assessment": {
            "label": judgment["label"],
            "grade": judgment["grade"],
            "reason": judgment["reason"],
        },
    }


def format_question(
    group: dict[str, Any],
    component_id: str,
    excluded: set[tuple[str, str]],
) -> dict[str, Any]:
    queries, query_by_normalized = dedupe_queries(group)
    evidence: dict[str, list[dict[str, Any]]] = {tier: [] for tier in TIERS}
    settled_grades = []
    for tier in TIERS:
        for verse in group["ayat"][tier]:
            disputed = (group["id"], verse["ayah"]) in excluded
            judgments = [
                format_judgment(judgment, query_by_normalized)
                for judgment in verse["judgments"]
            ]
            status = {"label_status": "disputed" if disputed else "settled_model_review"}
            if disputed:
                status["provisional_grade"] = verse["grade"]
            else:
                settled_grades.append(GRADES[tier])
            evidence[tier].append(
                {
                    "ayah": verse["ayah"],
                    "text": verse["text"],
                    "grade": None if disputed else GRADES[tier],
                    "status": status,
                    "judgments": judgments,
                }
            )

    has_positive = any(grade > 0 for grade in settled_grades)
    has_direct = any(grade == 3 for grade in settled_grades)
    if has_direct:
        stratum = "direct"
    elif any(grade == 2 for grade in settled_grades):
        stratum = "partial"
    elif any(grade == 1 for grade in settled_grades):
        stratum = "background"
    else:
        stratum = "none"
    return {
        "group_id": group["id"],
        "component_id": component_id,
        "provenance": {
            "origin_splits": group["source_splits"],
            "missing_reference_records": group["missing_reference_records"],
        },
        "queries": queries,
        "evidence": evidence,
        "label_summary": {"best_settled_label": stratum},
    }


def format_qrels(question: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "group_id": question["group_id"],
            "ayah": verse["ayah"],
            "label": tier,
            "grade": verse["grade"],
        }
        for tier in TIERS
        for verse in question["evidence"][tier]
        if verse["grade"] is not None
    ]


def format_candidates(group: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "group_id": group["id"],
            "ayah": candidate["ayah"],
            "text": candidate["text"],
        }
        for candidate in group["random_negative_candidates"]
    ]


def format_conflicts(
    conflicts: list[dict[str, Any]], ownership: dict[str, str]
) -> list[dict[str, Any]]:
    output = []
    for conflict in conflicts:
        labels = [
            {"label": label, "grade": GRADES[label]}
            for label in conflict["labels"]
        ]
        labels.sort(key=lambda item: (-item["grade"], item["label"]))
        output.append(
            {
                "split": ownership[conflict["group_id"]],
                "group_id": conflict["group_id"],
                "ayah": conflict["ayah"],
                "labels": labels,
                "provisional_grade": max(item["grade"] for item in labels),
            }
        )
    return output


def build_exclusions(
    groups: list[dict[str, Any]], ownership: dict[str, str]
) -> list[dict[str, Any]]:
    question_to_group = {
        normalize_text(query): group["id"]
        for group in groups
        for query in group["queries"]
    }
    exclusions = []
    for record in read_jsonl(DATA / "all_pairs_reviewed.jsonl"):
        reasons = []
        if record["review"]["relevance"] == "unresolved":
            reasons.append("unresolved_review")
        if not record.get("ayahs"):
            reasons.append("missing_verified_reference")
        if not reasons:
            continue
        group_id = question_to_group.get(normalize_text(record["question"]))
        exclusions.append(
            {
                "source_id": record["id"],
                "origin_split": record["split"],
                "group_id": group_id,
                "split": ownership.get(group_id),
                "question": record["question"],
                "source": {
                    "reference": record.get("source_reference"),
                    "quoted_text": record.get("quoted_text"),
                    "review_reason": record["review"].get("reason"),
                },
                "reasons": reasons,
            }
        )
    missing_ids = {
        source_id for group in groups for source_id in group["missing_reference_records"]
    }
    exclusion_ids = {row["source_id"] for row in exclusions}
    if not missing_ids.issubset(exclusion_ids):
        raise ValueError(f"Missing-reference records absent from exclusions: {missing_ids - exclusion_ids}")
    return exclusions


def build_similarity_audit(
    groups: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    decisions: dict[int, dict[str, Any]],
    ownership: dict[str, str],
    component_by_group: dict[str, str],
) -> list[dict[str, Any]]:
    question_to_group = {
        normalize_text(query): group["id"]
        for group in groups
        for query in group["queries"]
    }
    questions = read_json(DATA / "group_questions.json")
    audit = []
    for index, candidate in enumerate(candidates):
        group_a = question_to_group[normalize_text(questions[candidate["a"]])]
        group_b = question_to_group[normalize_text(questions[candidate["b"]])]
        approved = decisions[index]["merge"]
        same_component = component_by_group[group_a] == component_by_group[group_b]
        cross_split = ownership[group_a] != ownership[group_b]
        if approved and not same_component:
            raise ValueError("Approved equivalence edge crosses split components")
        audit.append(
            {
                "candidate_index": index,
                "pair": {
                    "question_a": candidate["question_a"],
                    "question_b": candidate["question_b"],
                    "group_a": group_a,
                    "group_b": group_b,
                },
                "similarity": {"cosine": candidate["cosine"]},
                "review": {
                    "approved_equivalence": approved,
                    "reason": decisions[index]["reason"],
                },
                "split_check": {
                    "same_component": same_component,
                    "cross_split": cross_split,
                },
            }
        )
    return audit


def load_corpus(path: Path) -> dict[str, str]:
    with sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True) as connection:
        rows = connection.execute("select sura, ayah, text from verses").fetchall()
    corpus = {f"{surah}:{ayah}": text for surah, ayah, text in rows}
    if len(corpus) != len(rows):
        raise ValueError("Translation corpus contains duplicate ayah references")
    return corpus


def validate_corpus(
    corpus: dict[str, str],
    questions: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> None:
    for question in questions:
        for verses in question["evidence"].values():
            for verse in verses:
                ayah_parts(verse["ayah"])
                if corpus.get(verse["ayah"]) != verse["text"]:
                    raise ValueError(f"Verse text does not match corpus: {verse['ayah']}")
    for candidate in candidates:
        ayah_parts(candidate["ayah"])
        if corpus.get(candidate["ayah"]) != candidate["text"]:
            raise ValueError(f"Candidate text does not match corpus: {candidate['ayah']}")


def source_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PACKAGE.parents[2], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def split_stats(
    questions: list[dict[str, Any]],
    qrels: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "question_groups": len(questions),
        "query_wordings": sum(len(question["queries"]) for question in questions),
        "judged_pairs": sum(
            len(verses)
            for question in questions
            for verses in question["evidence"].values()
        ),
        "scored_pairs": len(qrels),
        "disputed_pairs": sum(
            verse["grade"] is None
            for question in questions
            for verses in question["evidence"].values()
            for verse in verses
        ),
        "unjudged_candidates": len(candidates),
        "direct_question_groups": sum(
            question["label_summary"]["best_settled_label"] == "direct"
            for question in questions
        ),
        "positive_question_groups": sum(
            question["label_summary"]["best_settled_label"] != "none"
            for question in questions
        ),
        "groups_by_label_stratum": dict(
            Counter(question["label_summary"]["best_settled_label"] for question in questions)
        ),
        "groups_by_origin_profile": dict(
            Counter(origin_profile(question["provenance"]["origin_splits"]) for question in questions)
        ),
        "compound_query_groups": sum(
            any(
                query["stats"]["compound_status"] == "review_required"
                for query in question["queries"]
            )
            for question in questions
        ),
    }


def output_manifest_entry(path: Path, records: int | None = None) -> dict[str, Any]:
    entry: dict[str, Any] = {"sha256": file_sha256(path), "bytes": path.stat().st_size}
    if records is not None:
        entry["records"] = records
    return entry


def build_manifest(
    questions_by_split: dict[str, list[dict[str, Any]]],
    qrels_by_split: dict[str, list[dict[str, Any]]],
    candidates_by_split: dict[str, list[dict[str, Any]]],
    conflicts: list[dict[str, Any]],
    components: list[dict[str, Any]],
    similarity_audit: list[dict[str, Any]],
    exclusions: list[dict[str, Any]],
    corpus: dict[str, str],
    corpus_path: Path,
) -> dict[str, Any]:
    all_questions = [question for split in SPLITS for question in questions_by_split[split]]
    all_qrels = [qrel for split in SPLITS for qrel in qrels_by_split[split]]
    all_candidates = [candidate for split in SPLITS for candidate in candidates_by_split[split]]
    grade_counts = Counter(qrel["grade"] for qrel in all_qrels)
    class_weights = {
        str(grade): round(len(all_qrels) / (len(grade_counts) * count), 6)
        for grade, count in sorted(grade_counts.items())
    }
    outputs: dict[str, Any] = {}
    for split in SPLITS:
        for name, rows in (
            ("questions.jsonl", questions_by_split[split]),
            ("grades.jsonl", qrels_by_split[split]),
            ("unjudged_candidates.jsonl", candidates_by_split[split]),
        ):
            relative = f"{split}/{name}"
            outputs[relative] = output_manifest_entry(PACKAGE / relative, len(rows))
    for name, rows in (
        ("conflicts_excluded_from_grades.jsonl", conflicts),
        ("split_components.jsonl", components),
        ("similarity_audit.jsonl", similarity_audit),
        ("exclusions.jsonl", exclusions),
    ):
        outputs[name] = output_manifest_entry(PACKAGE / name, len(rows))

    component_sizes = Counter(len(component["group_ids"]) for component in components)
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_version": "2.0.0",
        "release_status": "provisional",
        "benchmark_grade": False,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": {
            "path": "create_splits.py",
            "sha256": file_sha256(Path(__file__)),
            "python_version": platform.python_version(),
            "source_commit": source_commit(),
        },
        "hash_policy": {
            "algorithm": "sha256",
            "source_inputs": "parsed canonical JSON with sorted keys and LF record separators",
            "outputs": "raw UTF-8 bytes with LF line endings",
        },
        "source_inputs": {
            name: semantic_sha256(DATA / name) for name in SOURCE_INPUTS
        },
        "outputs": outputs,
        "counts": {
            "question_groups": len(all_questions),
            "normalized_query_wordings": sum(
                len(question["queries"]) for question in all_questions
            ),
            "multi_query_groups": sum(
                len(question["queries"]) > 1 for question in all_questions
            ),
            "judged_pairs": sum(
                len(verses)
                for question in all_questions
                for verses in question["evidence"].values()
            ),
            "scored_pairs": len(all_qrels),
            "disputed_pairs": len(conflicts),
            "unjudged_candidates": len(all_candidates),
            "components": len(components),
            "similarity_candidates": len(similarity_audit),
            "approved_equivalence_edges": sum(
                row["review"]["approved_equivalence"] for row in similarity_audit
            ),
            "cross_split_rejected_candidates": sum(
                row["split_check"]["cross_split"]
                and not row["review"]["approved_equivalence"]
                for row in similarity_audit
            ),
            "exclusions": len(exclusions),
        },
        "splits": {
            split: split_stats(
                questions_by_split[split], qrels_by_split[split], candidates_by_split[split]
            )
            for split in SPLITS
        },
        "grade_distribution": {str(grade): grade_counts[grade] for grade in sorted(grade_counts)},
        "class_weights_balanced": class_weights,
        "component_size_distribution": {
            str(size): count for size, count in sorted(component_sizes.items())
        },
        "split_policy": {
            "ratios": RATIOS,
            "random_seed": SEED,
            "constraints": [
                "existing_question_groups_are_indivisible",
                "approved_equivalence_edges_are_kept_together",
                "normalized_exact_duplicate_questions_are_kept_together",
                "components_containing_a_group_without_a_settled_positive_go_to_train",
            ],
            "stratification": ["origin_profile", "best_settled_label"],
            "allocation": "largest_first_largest_normalized_remaining_deficit",
            "similarity_candidates_are_diagnostic_only": True,
        },
        "similarity_model": {
            "name": SIMILARITY_MODEL,
            "revision": SIMILARITY_REVISION,
            "candidate_threshold": SIMILARITY_THRESHOLD,
        },
        "translation": {
            "id": "Aayatun English Quran translation",
            "source_table": "verses(sura, ayah, text)",
            "corpus_path": corpus_path.name,
            "corpus_sha256": hashlib.sha256(canonical_json_bytes(corpus)).hexdigest(),
            "verse_count": len(corpus),
            "text_policy": "raw_corpus_text",
        },
        "annotation": {
            "extraction_model": "gpt-4o",
            "review_method": "model_assisted",
            "review_model_revision": None,
            "human_verified": False,
            "test_set_blind": False,
        },
        "release_blockers": [
            "independent_human_or_scholarly_review_pending",
            "citation_corrections_not_independently_verified",
            "test_questions_seen_during_model_review",
            "pooled_relevance_judgments_incomplete",
            "source_reuse_rights_and_license_unresolved",
            "compound_queries_not_atomically_reviewed",
            "english_only_evaluation_scope",
        ],
    }


def build(corpus_path: Path) -> dict[str, Any]:
    groups = read_jsonl(DATA / "grouped_queries.jsonl")
    questions = read_json(DATA / "group_questions.json")
    candidates, decisions = load_review_decisions()
    conflicts = read_jsonl(DATA / "grouped_conflicts.jsonl")
    excluded = {(row["group_id"], row["ayah"]) for row in conflicts}
    if len(excluded) != len(conflicts):
        raise ValueError("Duplicate conflicting question-ayah grades")

    components, component_by_group = build_components(
        groups, questions, candidates, decisions, excluded
    )
    ownership = allocate_splits(components, excluded)
    groups_by_id = {group["id"]: group for group in groups}

    questions_by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    qrels_by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    candidates_by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    for group in groups:
        split = ownership[group["id"]]
        question = format_question(
            group, component_by_group[group["id"]], excluded
        )
        questions_by_split[split].append(question)
        qrels_by_split[split].extend(format_qrels(question))
        candidates_by_split[split].extend(format_candidates(group))

    for split in SPLITS:
        questions_by_split[split].sort(key=lambda question: question["group_id"])
        qrels_by_split[split].sort(key=lambda qrel: (qrel["group_id"], qrel["ayah"]))
        candidates_by_split[split].sort(key=lambda row: (row["group_id"], row["ayah"]))
        write_jsonl(PACKAGE / split / "questions.jsonl", questions_by_split[split])
        write_jsonl(PACKAGE / split / "grades.jsonl", qrels_by_split[split])
        write_jsonl(
            PACKAGE / split / "unjudged_candidates.jsonl", candidates_by_split[split]
        )

    component_records = []
    for component in components:
        group_ids = [group["id"] for group in component["groups"]]
        splits = {ownership[group_id] for group_id in group_ids}
        if len(splits) != 1:
            raise ValueError("Component crosses splits")
        component_records.append(
            {
                "component_id": component["component_id"],
                "split": next(iter(splits)),
                "group_ids": group_ids,
                "reason": component["reason"],
            }
        )
    component_records.sort(key=lambda component: component["component_id"])
    conflict_records = format_conflicts(conflicts, ownership)
    exclusion_records = build_exclusions(groups, ownership)
    audit_records = build_similarity_audit(
        groups, candidates, decisions, ownership, component_by_group
    )

    write_jsonl(PACKAGE / "conflicts_excluded_from_grades.jsonl", conflict_records)
    write_jsonl(PACKAGE / "split_components.jsonl", component_records)
    write_jsonl(PACKAGE / "similarity_audit.jsonl", audit_records)
    write_jsonl(PACKAGE / "exclusions.jsonl", exclusion_records)

    corpus = load_corpus(corpus_path)
    validate_corpus(
        corpus,
        [question for split in SPLITS for question in questions_by_split[split]],
        [candidate for split in SPLITS for candidate in candidates_by_split[split]],
    )
    manifest = build_manifest(
        questions_by_split,
        qrels_by_split,
        candidates_by_split,
        conflict_records,
        component_records,
        audit_records,
        exclusion_records,
        corpus,
        corpus_path,
    )
    write_json(PACKAGE / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path, help="Aayatun en_quran.db")
    args = parser.parse_args()
    manifest = build(args.corpus)
    print(json.dumps({"counts": manifest["counts"], "splits": manifest["splits"]}, indent=2))


if __name__ == "__main__":
    main()
