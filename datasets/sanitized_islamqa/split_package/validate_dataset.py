"""Validate the IslamQA split package, its manifest, and optional corpus."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sqlite3
from typing import Any

from jsonschema import Draft202012Validator

from dataset_common import (
    GRADES,
    PACKAGE,
    SCHEMA_VERSION,
    SPLITS,
    TIERS,
    ayah_parts,
    file_sha256,
    normalize_text,
    origin_profile,
    read_json,
    read_jsonl,
    semantic_sha256,
    text_sha256,
)


class DatasetValidationError(ValueError):
    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("\n".join(errors))


def _validator(schema: dict[str, Any], definition: str) -> Draft202012Validator:
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(
        {"$schema": schema["$schema"], "$ref": f"#/$defs/{definition}", "$defs": schema["$defs"]}
    )


def _schema_errors(
    validator: Draft202012Validator, rows: list[dict[str, Any]], label: str
) -> list[str]:
    errors = []
    for index, row in enumerate(rows):
        for error in validator.iter_errors(row):
            path = "/".join(str(part) for part in error.absolute_path)
            errors.append(f"{label}[{index}]{('/' + path) if path else ''}: {error.message}")
    return errors


def _load_corpus(path: Path) -> dict[str, str]:
    with sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True) as connection:
        rows = connection.execute("select sura, ayah, text from verses").fetchall()
    corpus = {f"{surah}:{ayah}": text for surah, ayah, text in rows}
    if len(corpus) != len(rows):
        raise DatasetValidationError(["Translation corpus contains duplicate ayah references"])
    return corpus


def _compute_split_stats(
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
            Counter(
                question["label_summary"]["best_settled_label"]
                for question in questions
            )
        ),
        "groups_by_origin_profile": dict(
            Counter(
                origin_profile(question["provenance"]["origin_splits"])
                for question in questions
            )
        ),
        "compound_query_groups": sum(
            any(
                query["stats"]["compound_status"] == "review_required"
                for query in question["queries"]
            )
            for question in questions
        ),
    }


def validate_package(
    package: Path = PACKAGE,
    corpus_path: Path | None = None,
    check_hashes: bool = True,
) -> dict[str, Any]:
    errors: list[str] = []
    required_files = [
        "manifest.json",
        "schema.json",
        "conflicts_excluded_from_grades.jsonl",
        "split_components.jsonl",
        "similarity_audit.jsonl",
        "exclusions.jsonl",
        *[
            f"{split}/{name}"
            for split in SPLITS
            for name in ("questions.jsonl", "grades.jsonl", "unjudged_candidates.jsonl")
        ],
    ]
    for relative in required_files:
        if not (package / relative).is_file():
            errors.append(f"Missing required file: {relative}")
    if errors:
        raise DatasetValidationError(errors)

    manifest = read_json(package / "manifest.json")
    schema = read_json(package / "schema.json")
    data = {
        "question": {
            split: read_jsonl(package / split / "questions.jsonl") for split in SPLITS
        },
        "qrel": {split: read_jsonl(package / split / "grades.jsonl") for split in SPLITS},
        "candidate": {
            split: read_jsonl(package / split / "unjudged_candidates.jsonl")
            for split in SPLITS
        },
        "conflict": read_jsonl(package / "conflicts_excluded_from_grades.jsonl"),
        "component": read_jsonl(package / "split_components.jsonl"),
        "similarityAudit": read_jsonl(package / "similarity_audit.jsonl"),
        "exclusion": read_jsonl(package / "exclusions.jsonl"),
    }
    for definition, rows in data.items():
        if isinstance(rows, dict):
            for split, split_rows in rows.items():
                errors.extend(
                    _schema_errors(_validator(schema, definition), split_rows, f"{split}/{definition}")
                )
        else:
            errors.extend(_schema_errors(_validator(schema, definition), rows, definition))

    questions = {
        question["group_id"]: question
        for split in SPLITS
        for question in data["question"][split]
    }
    split_by_group = {
        question["group_id"]: split
        for split in SPLITS
        for question in data["question"][split]
    }
    if len(questions) != sum(len(rows) for rows in data["question"].values()):
        errors.append("Group IDs are not globally unique")
    components = {row["component_id"]: row for row in data["component"]}
    if len(components) != len(data["component"]):
        errors.append("Component IDs are not unique")

    query_ids: set[str] = set()
    evidence_by_group: dict[str, dict[str, dict[str, Any]]] = {}
    disputed_by_group: dict[str, dict[str, dict[str, Any]]] = {}
    source_ids_by_query: dict[str, set[str]] = defaultdict(set)

    for split in SPLITS:
        for question in data["question"][split]:
            group_id = question["group_id"]
            component = components.get(question["component_id"])
            if component is None:
                errors.append(f"{group_id}: unknown component {question['component_id']}")
            elif group_id not in component["group_ids"] or component["split"] != split:
                errors.append(f"{group_id}: component membership/split mismatch")

            normalized_queries = [normalize_text(query["text"]) for query in question["queries"]]
            if len(set(normalized_queries)) != len(normalized_queries):
                errors.append(f"{group_id}: normalized duplicate queries remain")
            query_lookup = {query["query_id"]: query for query in question["queries"]}
            for query in question["queries"]:
                if query["query_id"] in query_ids:
                    errors.append(f"Duplicate query ID: {query['query_id']}")
                query_ids.add(query["query_id"])
                if not query["query_id"].startswith(f"{group_id}:q"):
                    errors.append(f"{query['query_id']}: query ID has the wrong group prefix")
                expected_compound = (
                    "review_required"
                    if query["stats"]["question_mark_count"] > 1
                    or query["stats"]["char_length"] > 1000
                    else "not_flagged"
                )
                if query["stats"]["compound_status"] != expected_compound:
                    errors.append(f"{query['query_id']}: compound_status does not match stats")
                if query["stats"]["char_length"] != len(query["text"]):
                    errors.append(f"{query['query_id']}: char_length is incorrect")
                if query["stats"]["question_mark_count"] != query["text"].count("?"):
                    errors.append(f"{query['query_id']}: question_mark_count is incorrect")

            evidence: dict[str, dict[str, Any]] = {}
            disputed: dict[str, dict[str, Any]] = {}
            settled_grades = []
            for tier in TIERS:
                for verse in question["evidence"][tier]:
                    ayah = verse["ayah"]
                    if ayah in evidence:
                        errors.append(f"{group_id}: duplicate evidence reference {ayah}")
                    evidence[ayah] = verse
                    try:
                        ayah_parts(ayah)
                    except ValueError as exc:
                        errors.append(f"{group_id}/{ayah}: {exc}")
                    judgment_labels = {
                        judgment["assessment"]["label"]
                        for judgment in verse["judgments"]
                    }
                    for judgment in verse["judgments"]:
                        query = query_lookup.get(judgment["query_id"])
                        if query is None:
                            errors.append(
                                f"{group_id}/{ayah}: judgment references unknown query {judgment['query_id']}"
                            )
                        elif normalize_text(query["text"]) != normalize_text(judgment["query"]):
                            errors.append(
                                f"{group_id}/{ayah}: judgment query does not match query ID"
                            )
                        if ayah not in judgment["reviewed_passage"]:
                            errors.append(
                                f"{group_id}/{ayah}: ayah absent from reviewed_passage"
                            )
                        if judgment["assessment"]["grade"] != GRADES[judgment["assessment"]["label"]]:
                            errors.append(
                                f"{group_id}/{ayah}: judgment grade does not match its label"
                            )
                        source_ids_by_query[judgment["query_id"]].add(judgment["source_id"])

                    if verse["status"]["label_status"] == "disputed":
                        disputed[ayah] = verse
                        if verse["grade"] is not None:
                            errors.append(f"{group_id}/{ayah}: disputed grade must be null")
                        expected_provisional = max(
                            judgment["assessment"]["grade"]
                            for judgment in verse["judgments"]
                        )
                        if verse["status"].get("provisional_grade") != expected_provisional:
                            errors.append(f"{group_id}/{ayah}: provisional grade is incorrect")
                        if len(judgment_labels) < 2:
                            errors.append(f"{group_id}/{ayah}: disputed evidence has one label")
                    else:
                        settled_grades.append(verse["grade"])
                        if verse["grade"] != GRADES[tier]:
                            errors.append(f"{group_id}/{ayah}: settled grade does not match tier")
                        if judgment_labels != {tier}:
                            errors.append(f"{group_id}/{ayah}: settled judgments disagree with tier")

            evidence_by_group[group_id] = evidence
            disputed_by_group[group_id] = disputed
            for query in question["queries"]:
                if set(query["source_ids"]) != source_ids_by_query[query["query_id"]]:
                    errors.append(f"{query['query_id']}: source_ids do not match judgments")

            has_positive = any(grade > 0 for grade in settled_grades)
            has_direct = any(grade == 3 for grade in settled_grades)
            expected_stratum = (
                "direct" if has_direct else
                "partial" if any(grade == 2 for grade in settled_grades) else
                "background" if any(grade == 1 for grade in settled_grades) else
                "none"
            )
            summary = question["label_summary"]
            if summary != {"best_settled_label": expected_stratum}:
                errors.append(f"{group_id}: label_summary is incorrect")
            if split in ("validation", "test") and not has_positive:
                errors.append(f"{group_id}: evaluation group has no settled positive")

    qrels_by_group: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for split in SPLITS:
        for qrel in data["qrel"][split]:
            group_id = qrel["group_id"]
            if group_id not in questions:
                errors.append(f"{group_id}/{qrel['ayah']}: qrel references unknown group")
                continue
            if split_by_group[group_id] != split:
                errors.append(f"{group_id}/{qrel['ayah']}: qrel is in the wrong split file")
            if qrel["ayah"] in qrels_by_group[group_id]:
                errors.append(f"{group_id}/{qrel['ayah']}: duplicate qrel")
            qrels_by_group[group_id][qrel["ayah"]] = qrel

    for group_id, question in questions.items():
        expected = {
            verse["ayah"]: (tier, verse)
            for tier in TIERS
            for verse in question["evidence"][tier]
            if verse["grade"] is not None
        }
        actual = qrels_by_group[group_id]
        if set(expected) != set(actual):
            errors.append(
                f"{group_id}: qrel keys differ from settled evidence; "
                f"missing={sorted(set(expected)-set(actual))[:3]}, "
                f"extra={sorted(set(actual)-set(expected))[:3]}"
            )
        for ayah in set(expected) & set(actual):
            tier, verse = expected[ayah]
            qrel = actual[ayah]
            if (qrel["grade"], qrel["label"]) != (verse["grade"], tier):
                errors.append(f"{group_id}/{ayah}: qrel value differs from evidence")

    candidate_keys: set[tuple[str, str]] = set()
    for split in SPLITS:
        for candidate in data["candidate"][split]:
            key = (candidate["group_id"], candidate["ayah"])
            if key in candidate_keys:
                errors.append(f"{key}: duplicate unjudged candidate")
            candidate_keys.add(key)
            if candidate["group_id"] not in questions:
                errors.append(f"{key}: candidate references unknown group")
            elif split_by_group[candidate["group_id"]] != split:
                errors.append(f"{key}: candidate is in the wrong split file")
            elif candidate["ayah"] in evidence_by_group[candidate["group_id"]]:
                errors.append(f"{key}: unjudged candidate overlaps judged evidence")
            try:
                ayah_parts(candidate["ayah"])
            except ValueError as exc:
                errors.append(f"{key}: {exc}")

    conflict_keys: set[tuple[str, str]] = set()
    for conflict in data["conflict"]:
        key = (conflict["group_id"], conflict["ayah"])
        if key in conflict_keys:
            errors.append(f"{key}: duplicate conflict")
        conflict_keys.add(key)
        verse = disputed_by_group.get(conflict["group_id"], {}).get(conflict["ayah"])
        if verse is None:
            errors.append(f"{key}: conflict has no matching disputed evidence")
            continue
        labels = {item["label"] for item in conflict["labels"]}
        judgment_labels = {
            judgment["assessment"]["label"] for judgment in verse["judgments"]
        }
        if labels != judgment_labels:
            errors.append(f"{key}: conflict labels differ from judgments")
        if conflict["provisional_grade"] != max(
            item["grade"] for item in conflict["labels"]
        ):
            errors.append(f"{key}: conflict provisional grade is incorrect")
        if split_by_group[conflict["group_id"]] != conflict["split"]:
            errors.append(f"{key}: conflict split mismatch")
    expected_conflicts = {
        (group_id, ayah)
        for group_id, disputed in disputed_by_group.items()
        for ayah in disputed
    }
    if conflict_keys != expected_conflicts:
        errors.append("Conflict file does not exactly cover disputed evidence")

    component_owners: dict[str, str] = {}
    for component in data["component"]:
        for group_id in component["group_ids"]:
            if group_id in component_owners:
                errors.append(f"{group_id}: group appears in multiple components")
            component_owners[group_id] = component["component_id"]
            if group_id not in questions:
                errors.append(f"{component['component_id']}: unknown group {group_id}")
            elif questions[group_id]["component_id"] != component["component_id"]:
                errors.append(f"{group_id}: question/component ID mismatch")
            elif split_by_group[group_id] != component["split"]:
                errors.append(f"{group_id}: question/component split mismatch")
        has_nonpositive = any(
            questions[group_id]["label_summary"]["best_settled_label"] == "none"
            for group_id in component["group_ids"]
            if group_id in questions
        )
        expected_reason = (
            "contains_group_without_settled_positive"
            if has_nonpositive
            else "balanced_stratified_allocation"
        )
        if component["reason"] != expected_reason:
            errors.append(f"{component['component_id']}: allocation reason is incorrect")
        if has_nonpositive and component["split"] != "train":
            errors.append(f"{component['component_id']}: forced component is not in train")
    if set(component_owners) != set(questions):
        errors.append("Component membership does not cover every question group")

    audit_indices = set()
    for audit in data["similarityAudit"]:
        index = audit["candidate_index"]
        if index in audit_indices:
            errors.append(f"Duplicate similarity candidate index: {index}")
        audit_indices.add(index)
        pair = audit["pair"]
        review = audit["review"]
        split_check = audit["split_check"]
        if pair["group_a"] not in questions or pair["group_b"] not in questions:
            errors.append(f"Similarity candidate {index}: unknown group")
            continue
        if split_check["same_component"] != (
            questions[pair["group_a"]]["component_id"]
            == questions[pair["group_b"]]["component_id"]
        ):
            errors.append(f"Similarity candidate {index}: same_component flag is incorrect")
        if split_check["cross_split"] != (
            split_by_group[pair["group_a"]] != split_by_group[pair["group_b"]]
        ):
            errors.append(f"Similarity candidate {index}: cross_split flag is incorrect")
        if review["approved_equivalence"]:
            if not split_check["same_component"] or split_check["cross_split"]:
                errors.append(f"Similarity candidate {index}: approved edge crosses components")
    if audit_indices != set(range(len(data["similarityAudit"]))):
        errors.append("Similarity audit indices are not contiguous")

    exclusion_ids = set()
    for exclusion in data["exclusion"]:
        if exclusion["source_id"] in exclusion_ids:
            errors.append(f"Duplicate exclusion source ID: {exclusion['source_id']}")
        exclusion_ids.add(exclusion["source_id"])
        group_id = exclusion["group_id"]
        if group_id is None:
            if exclusion["split"] is not None:
                errors.append(f"{exclusion['source_id']}: split present without group")
        elif group_id not in questions:
            errors.append(f"{exclusion['source_id']}: unknown excluded group")
        elif exclusion["split"] != split_by_group[group_id]:
            errors.append(f"{exclusion['source_id']}: excluded split mismatch")
    missing_ids = {
        source_id
        for question in questions.values()
        for source_id in question["provenance"]["missing_reference_records"]
    }
    if not missing_ids.issubset(exclusion_ids):
        errors.append("Missing-reference records are absent from exclusions")

    all_questions = [questions[group_id] for group_id in sorted(questions)]
    all_qrels = [qrel for split in SPLITS for qrel in data["qrel"][split]]
    all_candidates = [row for split in SPLITS for row in data["candidate"][split]]
    grade_counts = Counter(qrel["grade"] for qrel in all_qrels)
    expected_counts = {
        "question_groups": len(all_questions),
        "normalized_query_wordings": sum(len(question["queries"]) for question in all_questions),
        "multi_query_groups": sum(len(question["queries"]) > 1 for question in all_questions),
        "judged_pairs": sum(
            len(verses)
            for question in all_questions
            for verses in question["evidence"].values()
        ),
        "scored_pairs": len(all_qrels),
        "disputed_pairs": len(data["conflict"]),
        "unjudged_candidates": len(all_candidates),
        "components": len(data["component"]),
        "similarity_candidates": len(data["similarityAudit"]),
        "approved_equivalence_edges": sum(
            row["review"]["approved_equivalence"] for row in data["similarityAudit"]
        ),
        "cross_split_rejected_candidates": sum(
            row["split_check"]["cross_split"]
            and not row["review"]["approved_equivalence"]
            for row in data["similarityAudit"]
        ),
        "exclusions": len(data["exclusion"]),
    }
    if manifest.get("counts") != expected_counts:
        errors.append("Manifest counts do not match package contents")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append("Manifest schema_version is incorrect")
    if manifest.get("release_status") != "provisional" or manifest.get("benchmark_grade") is not False:
        errors.append("Manifest must retain provisional/non-benchmark release status")
    if manifest.get("grade_distribution") != {
        str(grade): grade_counts[grade] for grade in sorted(grade_counts)
    }:
        errors.append("Manifest grade distribution is incorrect")
    for split in SPLITS:
        expected = _compute_split_stats(
            data["question"][split], data["qrel"][split], data["candidate"][split]
        )
        if manifest.get("splits", {}).get(split) != expected:
            errors.append(f"Manifest statistics for {split} are incorrect")

    if check_hashes:
        for name, expected_hash in manifest.get("source_inputs", {}).items():
            path = package.parent / name
            if not path.is_file():
                errors.append(f"Manifest source input is missing: {name}")
            elif semantic_sha256(path) != expected_hash:
                errors.append(f"Manifest source hash mismatch: {name}")
        for relative, entry in manifest.get("outputs", {}).items():
            path = package / relative
            if not path.is_file():
                errors.append(f"Manifest output is missing: {relative}")
                continue
            if file_sha256(path) != entry.get("sha256"):
                errors.append(f"Manifest output hash mismatch: {relative}")
            if path.suffix == ".jsonl":
                records = len(read_jsonl(path))
                if records != entry.get("records"):
                    errors.append(f"Manifest output record count mismatch: {relative}")
        generator = package / manifest.get("generator", {}).get("path", "create_splits.py")
        if generator.is_file():
            if text_sha256(generator) != manifest.get("generator", {}).get("sha256"):
                errors.append("Manifest generator hash mismatch")
        else:
            errors.append("Manifest generator file is missing")

    if corpus_path is not None:
        corpus = _load_corpus(corpus_path)
        for question in all_questions:
            for verses in question["evidence"].values():
                for verse in verses:
                    if corpus.get(verse["ayah"]) != verse["text"]:
                        errors.append(
                            f"{question['group_id']}/{verse['ayah']}: corpus text mismatch"
                        )
        for candidate in all_candidates:
            if corpus.get(candidate["ayah"]) != candidate["text"]:
                errors.append(
                    f"{candidate['group_id']}/{candidate['ayah']}: candidate corpus text mismatch"
                )
        translation = manifest.get("translation", {})
        if translation.get("verse_count") != len(corpus):
            errors.append("Manifest translation verse_count is incorrect")
        import hashlib
        from dataset_common import canonical_json_bytes

        corpus_hash = hashlib.sha256(canonical_json_bytes(corpus)).hexdigest()
        if translation.get("corpus_sha256") != corpus_hash:
            errors.append("Manifest translation corpus hash is incorrect")

    if errors:
        raise DatasetValidationError(errors)
    return {
        "valid": True,
        "counts": expected_counts,
        "splits": manifest["splits"],
        "errors": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=PACKAGE)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--check", action="store_true", help="Validate without modifying files")
    args = parser.parse_args()
    try:
        result = validate_package(args.package, args.corpus, check_hashes=True)
    except DatasetValidationError as exc:
        print(json.dumps({"valid": False, "errors": exc.errors}, indent=2))
        raise SystemExit(1)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
