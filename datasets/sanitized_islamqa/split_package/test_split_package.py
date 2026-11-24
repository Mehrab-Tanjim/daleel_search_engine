"""Synthetic regression tests for the IslamQA split package v2."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from build_review_pool import build_review_pool
from create_splits import (
    allocate_splits,
    build_components,
    dedupe_queries,
    format_conflicts,
    format_candidates,
    format_qrels,
    format_question,
    group_label_stratum,
    split_stats,
)
from dataset_common import (
    GRADES,
    PACKAGE,
    SCHEMA_VERSION,
    SPLITS,
    TIERS,
    file_sha256,
    semantic_sha256,
    stable_id,
    text_sha256,
    write_json,
    write_jsonl,
)
from evaluate_search import (
    bootstrap_intervals,
    dcg,
    evaluate_run,
    load_ranking,
    paired_comparison,
    score_group,
)
from load_dataset import Dataset
from validate_dataset import DatasetValidationError, validate_package

TRAIN_GROUP = "0000000000000001"
VALIDATION_GROUP = "0000000000000002"


def make_group(
    group_id: str,
    query: str,
    tier: str,
    ayah: str,
    *,
    origin: str = "train",
    judgments: list[dict] | None = None,
) -> dict:
    if judgments is None:
        judgments = [
            {
                "source_id": f"{origin}:{group_id}:0",
                "query": query,
                "source_reference": ayah,
                "reviewed_passage": [ayah],
                "label": tier,
                "grade": GRADES[tier],
                "reason": "Synthetic test judgment.",
            }
        ]
    ayat = {name: [] for name in TIERS}
    ayat[tier].append(
        {
            "ayah": ayah,
            "text": f"Synthetic text for {ayah}",
            "grade": GRADES[tier],
            "judgments": judgments,
        }
    )
    return {
        "id": group_id,
        "queries": [query],
        "source_splits": [origin],
        "missing_reference_records": [],
        "ayat": ayat,
        "random_negative_candidates": [],
    }


def make_minimal_package(root: Path) -> tuple[Dataset, Path]:
    shutil.copyfile(PACKAGE / "schema.json", root / "schema.json")
    train_group = make_group(
        TRAIN_GROUP, "Unrelated synthetic question?", "unrelated", "1:1"
    )
    validation_group = make_group(
        VALIDATION_GROUP,
        "Direct synthetic question?",
        "direct",
        "1:2",
        origin="benchmark",
    )
    train_question = format_question(
        train_group, stable_id("cmp_", TRAIN_GROUP), set()
    )
    validation_question = format_question(
        validation_group, stable_id("cmp_", VALIDATION_GROUP), set()
    )
    questions = {"train": [train_question], "validation": [validation_question], "test": []}
    qrels = {
        "train": format_qrels(train_question),
        "validation": format_qrels(validation_question),
        "test": [],
    }
    candidates = {split: [] for split in SPLITS}
    for split in SPLITS:
        write_jsonl(root / split / "questions.jsonl", questions[split])
        write_jsonl(root / split / "grades.jsonl", qrels[split])
        write_jsonl(root / split / "unjudged_candidates.jsonl", candidates[split])

    components = [
        {
            "component_id": train_question["component_id"],
            "split": "train",
            "group_ids": [TRAIN_GROUP],
            "reason": "contains_group_without_settled_positive",
        },
        {
            "component_id": validation_question["component_id"],
            "split": "validation",
            "group_ids": [VALIDATION_GROUP],
            "reason": "balanced_stratified_allocation",
        },
    ]
    write_jsonl(root / "split_components.jsonl", components)
    write_jsonl(root / "conflicts_excluded_from_grades.jsonl", [])
    write_jsonl(root / "similarity_audit.jsonl", [])
    write_jsonl(root / "exclusions.jsonl", [])
    all_questions = [train_question, validation_question]
    all_qrels = qrels["train"] + qrels["validation"]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "release_status": "provisional",
        "benchmark_grade": False,
        "source_inputs": {},
        "outputs": {},
        "counts": {
            "question_groups": 2,
            "normalized_query_wordings": 2,
            "multi_query_groups": 0,
            "judged_pairs": 2,
            "scored_pairs": 2,
            "disputed_pairs": 0,
            "unjudged_candidates": 0,
            "components": 2,
            "similarity_candidates": 0,
            "approved_equivalence_edges": 0,
            "cross_split_rejected_candidates": 0,
            "exclusions": 0,
        },
        "splits": {
            split: split_stats(questions[split], qrels[split], candidates[split])
            for split in SPLITS
        },
        "grade_distribution": {
            str(grade): sum(row["grade"] == grade for row in all_qrels)
            for grade in sorted({row["grade"] for row in all_qrels})
        },
    }
    write_json(root / "manifest.json", manifest)
    return Dataset.load(root, validate=False), root


class SplitLogicTests(unittest.TestCase):
    def test_only_approved_edges_join_components(self) -> None:
        groups = [
            make_group("0000000000000010", "First question?", "direct", "1:1"),
            make_group("0000000000000011", "Second question?", "partial", "1:2"),
        ]
        candidates = [
            {
                "a": 0,
                "b": 1,
                "cosine": 0.97,
                "question_a": "First question?",
                "question_b": "Second question?",
            }
        ]
        rejected = {0: {"merge": False, "reason": "Different intent."}}
        approved = {0: {"merge": True, "reason": "Equivalent intent."}}

        rejected_components, _ = build_components(
            groups, [group["queries"][0] for group in groups], candidates, rejected, set()
        )
        approved_components, ownership = build_components(
            groups, [group["queries"][0] for group in groups], candidates, approved, set()
        )
        self.assertEqual(len(rejected_components), 2)
        self.assertEqual(len(approved_components), 1)
        self.assertEqual(len(set(ownership.values())), 1)

    def test_normalized_query_variants_are_deduplicated(self) -> None:
        query = "Who goes to heaven?"
        variant = "  who   GOES to heaven?  "
        group = make_group("0000000000000020", query, "direct", "2:255")
        group["queries"].append(variant)
        group["ayat"]["direct"][0]["judgments"].append(
            {
                "source_id": "train:variant:1",
                "query": variant,
                "source_reference": "2:255",
                "reviewed_passage": ["2:255"],
                "label": "direct",
                "grade": 3,
                "reason": "Second source for the same wording.",
            }
        )
        queries, _ = dedupe_queries(group)
        self.assertEqual(len(queries), 1)
        self.assertEqual(queries[0]["text"], query)
        self.assertNotIn("source_ids", queries[0])
        question = format_question(
            group, stable_id("cmp_", group["id"]), set()
        )
        judgments = question["evidence"]["direct"][0]["judgments"]
        self.assertNotIn("source_query", judgments[0])
        self.assertEqual(judgments[1]["source_query"], variant)
        self.assertEqual(
            [judgment["source_id"] for judgment in judgments],
            ["train:0000000000000020:0", "train:variant:1"],
        )
        self.assertTrue(all("grade" not in item["assessment"] for item in judgments))

    def test_conflicts_have_null_grades_and_no_qrels(self) -> None:
        group_id = "0000000000000030"
        query = "Conflicted synthetic question?"
        judgments = [
            {
                "source_id": "train:conflict:0",
                "query": query,
                "source_reference": "3:1",
                "reviewed_passage": ["3:1"],
                "label": "direct",
                "grade": 3,
                "reason": "Direct interpretation.",
            },
            {
                "source_id": "train:conflict:1",
                "query": query,
                "source_reference": "3:1",
                "reviewed_passage": ["3:1"],
                "label": "partial",
                "grade": 2,
                "reason": "Partial interpretation.",
            },
        ]
        group = make_group(
            group_id,
            query,
            "direct",
            "3:1",
            judgments=judgments,
        )
        question = format_question(
            group, stable_id("cmp_", group_id), {(group_id, "3:1")}
        )
        verse = question["evidence"]["direct"][0]
        self.assertIsNone(verse["grade"])
        self.assertEqual(
            verse["status"],
            {"label_status": "disputed", "provisional_grade": 3},
        )
        self.assertEqual(format_qrels(question), [])
        self.assertEqual(
            [item["assessment"]["label"] for item in verse["judgments"]],
            ["direct", "partial"],
        )
        self.assertTrue(all("grade" not in item["assessment"] for item in verse["judgments"]))
        self.assertTrue(all("source_query" not in item for item in verse["judgments"]))
        self.assertEqual(
            question["label_summary"], {"best_settled_label": "none"}
        )
        conflicts = format_conflicts(
            [
                {
                    "group_id": group_id,
                    "ayah": "3:1",
                    "labels": ["direct", "partial"],
                }
            ],
            {group_id: "train"},
        )
        self.assertEqual(conflicts[0]["provisional_grade"], 3)
        self.assertNotIn("resolution", conflicts[0])

    def test_no_positive_components_are_forced_to_train(self) -> None:
        groups = [
            make_group("0000000000000040", "No positive?", "unrelated", "4:1"),
            make_group("0000000000000041", "Has positive?", "direct", "4:2"),
        ]
        components, _ = build_components(
            groups,
            [group["queries"][0] for group in groups],
            [],
            {},
            set(),
        )
        no_positive = next(
            component
            for component in components
            if component["groups"][0]["id"] == groups[0]["id"]
        )
        self.assertTrue(no_positive["forced_to_train"])
        self.assertEqual(
            no_positive["reason"], "contains_group_without_settled_positive"
        )
        ownership = allocate_splits(components, set())
        self.assertEqual(ownership[groups[0]["id"]], "train")

    def test_component_integrity_and_stratified_allocation(self) -> None:
        groups = []
        for index in range(100):
            group_id = f"{index + 1:016x}"
            tier = ("direct", "partial", "background")[index % 3]
            groups.append(
                make_group(
                    group_id,
                    f"Synthetic question number {index}?",
                    tier,
                    f"1:{index % 286 + 1}",
                    origin="train" if index % 2 == 0 else "benchmark",
                )
            )
        components, component_by_group = build_components(
            groups,
            [group["queries"][0] for group in groups],
            [],
            {},
            set(),
        )
        ownership = allocate_splits(components, set())
        counts = Counter(ownership.values())
        self.assertEqual(sum(counts.values()), 100)
        self.assertLessEqual(abs(counts["train"] - 65), 3)
        self.assertLessEqual(abs(counts["validation"] - 15), 3)
        self.assertLessEqual(abs(counts["test"] - 20), 3)
        groups_by_id = {group["id"]: group for group in groups}
        for component in components:
            splits = {ownership[group["id"]] for group in component["groups"]}
            self.assertEqual(len(splits), 1)
            self.assertEqual(
                component_by_group[component["groups"][0]["id"]],
                stable_id("cmp_", *(group["id"] for group in component["groups"])),
            )
        for group_id, split in ownership.items():
            if split in ("validation", "test"):
                self.assertNotEqual(
                    group_label_stratum(groups_by_id[group_id], set()), "none"
                )


class PackageAndHashTests(unittest.TestCase):
    def test_semantic_hashes_ignore_line_endings_and_whitespace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lf = root / "lf.jsonl"
            crlf = root / "crlf.jsonl"
            rows = [{"b": 2, "a": 1}, {"c": "text"}]
            lf.write_bytes(
                b"".join(
                    json.dumps(row, sort_keys=True).encode("utf8") + b"\n"
                    for row in rows
                )
            )
            crlf.write_bytes(
                b"".join(
                    json.dumps(row, sort_keys=True).encode("utf8") + b"\r\n"
                    for row in rows
                )
            )
            self.assertEqual(semantic_sha256(lf), semantic_sha256(crlf))
            self.assertNotEqual(file_sha256(lf), file_sha256(crlf))

            compact = root / "compact.json"
            pretty = root / "pretty.json"
            compact.write_text('{"a":1,"b":[2,3]}', encoding="utf8")
            pretty.write_text('{\n  "b": [2, 3],\n  "a": 1\n}\n', encoding="utf8")
            self.assertEqual(semantic_sha256(compact), semantic_sha256(pretty))

    def test_generator_hash_ignores_line_endings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lf = root / "generator_lf.py"
            crlf = root / "generator_crlf.py"
            lf.write_bytes(b"print('hello')\n")
            crlf.write_bytes(b"print('hello')\r\n")
            self.assertEqual(text_sha256(lf), text_sha256(crlf))
            self.assertNotEqual(file_sha256(lf), file_sha256(crlf))

    def test_minimal_package_validates_and_loader_exposes_join_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset, root = make_minimal_package(Path(directory))
            result = validate_package(root, check_hashes=False)
            self.assertTrue(result["valid"])
            self.assertEqual(dataset.split_for_group(VALIDATION_GROUP), "validation")
            self.assertEqual(
                dataset.splits["validation"].qrels[VALIDATION_GROUP], {"1:2": 3}
            )
            self.assertEqual(
                dataset.judged_pairs_for(VALIDATION_GROUP), frozenset({"1:2"})
            )
            self.assertNotIn("split", dataset.questions[VALIDATION_GROUP])
            candidate_group = make_group(
                TRAIN_GROUP, "Candidate question?", "unrelated", "1:1"
            )
            candidate_group["random_negative_candidates"] = [
                {"ayah": "1:3", "text": "Unjudged candidate"}
            ]
            self.assertEqual(
                format_candidates(candidate_group),
                [
                    {
                        "group_id": TRAIN_GROUP,
                        "ayah": "1:3",
                        "text": "Unjudged candidate",
                    }
                ],
            )

    def test_schema_and_referential_integrity_failures_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, root = make_minimal_package(Path(directory))
            questions_path = root / "validation" / "questions.jsonl"
            rows = [json.loads(line) for line in questions_path.read_text(encoding="utf8").splitlines()]
            rows[0]["relevance_order"] = 1
            write_jsonl(questions_path, rows)
            with self.assertRaises(DatasetValidationError) as caught:
                validate_package(root, check_hashes=False)
            self.assertTrue(any("Additional properties" in error for error in caught.exception.errors))

        with tempfile.TemporaryDirectory() as directory:
            _, root = make_minimal_package(Path(directory))
            qrels_path = root / "validation" / "grades.jsonl"
            rows = [json.loads(line) for line in qrels_path.read_text(encoding="utf8").splitlines()]
            rows[0]["grade"] = 2
            write_jsonl(qrels_path, rows)
            with self.assertRaises(DatasetValidationError) as caught:
                validate_package(root, check_hashes=False)
            self.assertTrue(any("qrel value differs" in error for error in caught.exception.errors))

    def test_ranking_loader_rejects_duplicates_and_unknown_groups(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ranking.jsonl"
            write_jsonl(
                path,
                [
                    {"group_id": VALIDATION_GROUP, "ranking": ["1:1", "1:1"]},
                ],
            )
            with self.assertRaisesRegex(ValueError, "duplicate ayah"):
                load_ranking(path, {VALIDATION_GROUP})
            write_jsonl(
                path,
                [{"group_id": "0000000000000099", "ranking": ["1:1"]}],
            )
            with self.assertRaisesRegex(ValueError, "unknown group"):
                load_ranking(path, {VALIDATION_GROUP})
            path.write_text("[]\n", encoding="utf8")
            with self.assertRaisesRegex(ValueError, "expected group_id and ranking"):
                load_ranking(path, {VALIDATION_GROUP})
            write_jsonl(
                path,
                [{"group_id": VALIDATION_GROUP, "ranking": [1]}],
            )
            with self.assertRaisesRegex(ValueError, "ayah strings"):
                load_ranking(path, {VALIDATION_GROUP})


class EvaluationTests(unittest.TestCase):
    def test_metric_definitions_and_unjudged_rank_positions(self) -> None:
        qrels = {"1:1": 3, "1:2": 2, "1:3": 1, "1:4": 0}
        scores = score_group(["9:9", "1:1", "1:2", "1:3", "1:4"], qrels)
        self.assertEqual(scores["direct_hit@5"], 1.0)
        self.assertEqual(scores["useful_hit@5"], 1.0)
        self.assertEqual(scores["positive_hit@5"], 1.0)
        self.assertEqual(scores["recall@5"], 1.0)
        self.assertEqual(scores["recall@10"], 1.0)
        self.assertEqual(scores["mrr@10"], 0.5)
        expected_ndcg = dcg([0, 3, 2, 1, 0], 10) / dcg([3, 2, 1, 0], 10)
        self.assertAlmostEqual(scores["ndcg@10"], expected_ndcg)
        empty = score_group(["9:9"], qrels)
        self.assertTrue(all(value == 0 for value in empty.values()))
        with self.assertRaisesRegex(ValueError, "no settled positive"):
            score_group([], {"1:1": 0})

    def test_bootstrap_intervals_are_deterministic(self) -> None:
        scores = {
            "a": {"direct_hit@5": 1.0},
            "b": {"direct_hit@5": 0.0},
            "c": {"direct_hit@5": 1.0},
        }
        full = {
            group_id: {
                metric: score["direct_hit@5"]
                for metric in (
                    "direct_hit@5",
                    "useful_hit@5",
                    "positive_hit@5",
                    "recall@5",
                    "recall@10",
                    "mrr@10",
                    "ndcg@10",
                )
            }
            for group_id, score in scores.items()
        }
        first = bootstrap_intervals(full, samples=100, seed=7)
        second = bootstrap_intervals(full, samples=100, seed=7)
        self.assertEqual(first, second)
        lower, upper = first["direct_hit@5"]
        self.assertLessEqual(lower, upper)
        self.assertEqual(bootstrap_intervals(full, samples=0, seed=7), {})
        with self.assertRaisesRegex(ValueError, "non-negative"):
            bootstrap_intervals(full, samples=-1, seed=7)

    def test_paired_wins_losses_ties_and_delta_interval(self) -> None:
        baseline = {
            "a": {"ndcg@10": 0.5},
            "b": {"ndcg@10": 0.5},
            "c": {"ndcg@10": 0.5},
        }
        candidate = {
            "a": {"ndcg@10": 0.8},
            "b": {"ndcg@10": 0.2},
            "c": {"ndcg@10": 0.5},
        }
        result = paired_comparison(
            "baseline",
            "candidate",
            baseline,
            candidate,
            "ndcg@10",
            bootstrap_samples=100,
            seed=11,
        )
        self.assertEqual((result["wins"], result["losses"], result["ties"]), (1, 1, 1))
        self.assertAlmostEqual(result["delta_mean"], 0.0)
        lower, upper = result["delta_ci95"]
        self.assertLess(lower, 0)
        self.assertGreater(upper, 0)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            paired_comparison(
                "baseline",
                "candidate",
                baseline,
                candidate,
                "ndcg@10",
                bootstrap_samples=-1,
            )

    def test_run_report_includes_coverage_strata_and_missing_groups(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset, _ = make_minimal_package(Path(directory))
            report, scores = evaluate_run(
                dataset,
                ["validation"],
                {VALIDATION_GROUP: ["1:1", "1:2"]},
                bootstrap_samples=50,
                seed=3,
            )
            self.assertEqual(report["eligible_groups"], 1)
            self.assertEqual(report["result_coverage"]["settled_judged_results"], 1)
            self.assertEqual(report["result_coverage"]["unjudged_results"], 1)
            self.assertIn("direct", report["by_stratum"])
            self.assertEqual(scores[VALIDATION_GROUP]["mrr@10"], 0.5)

            missing, _ = evaluate_run(
                dataset,
                ["validation"],
                {},
                bootstrap_samples=0,
            )
            self.assertEqual(missing["missing_eligible_groups"], 1)
            self.assertEqual(missing["metrics"]["ndcg@10"]["mean"], 0.0)


class ReviewPoolTests(unittest.TestCase):
    def test_review_pool_deduplicates_and_excludes_judged_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset, _ = make_minimal_package(Path(directory))
            runs = {
                "run_a": {VALIDATION_GROUP: ["1:2", "1:1", "1:3"]},
                "run_b": {VALIDATION_GROUP: ["1:1", "1:3"]},
            }
            corpus = {"1:1": "First", "1:2": "Judged", "1:3": "Third"}
            rows = build_review_pool(
                dataset,
                runs,
                ["validation"],
                top_k=3,
                corpus=corpus,
            )
            self.assertEqual([row["ayah"] for row in rows], ["1:1", "1:3"])
            self.assertEqual(
                rows[0]["sources"],
                [{"run": "run_a", "rank": 2}, {"run": "run_b", "rank": 1}],
            )
            self.assertEqual(rows[0]["queries"][0]["query_id"], f"{VALIDATION_GROUP}:q0")
            self.assertEqual(rows[0]["text"], "First")
            self.assertNotIn("1:2", [row["ayah"] for row in rows])


if __name__ == "__main__":
    unittest.main()
