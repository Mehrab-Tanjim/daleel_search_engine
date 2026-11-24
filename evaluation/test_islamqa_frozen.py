from __future__ import annotations

import copy
import hashlib
import os
import subprocess
import sys
import unittest
from unittest import mock

import islamqa_frozen as frozen


class FrozenDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = frozen.load_cases(frozen.DEFAULT_PACKAGE)

    def test_v3_package_hashes_and_preserved_splits(self):
        manifest = frozen.verify_v3_package(frozen.DEFAULT_PACKAGE)
        self.assertEqual(manifest["dataset_version"], "3.0.0")
        self.assertEqual(manifest["train_groups"], 1086)
        self.assertFalse(manifest["test_set_blind"])

    def test_v3_hash_mutation_is_rejected(self):
        with mock.patch.object(frozen, "sha256_file", return_value="0" * 64):
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                frozen.verify_v3_package(frozen.DEFAULT_PACKAGE)

    def test_original_query_denominators_are_preserved(self):
        counts = {
            "queries": len(self.cases),
            "groups": len({case["group_id"] for case in self.cases}),
            "useful": sum(
                any(grade >= 2 for grade in case["qrels"].values())
                for case in self.cases
            ),
            "direct": sum(
                case["best_settled_label"] == "direct" for case in self.cases
            ),
        }
        self.assertEqual(
            counts,
            {"queries": 345, "groups": 341, "useful": 294, "direct": 54},
        )

    def test_fixture_is_byte_identical_to_historical_pr2(self):
        digest = hashlib.sha256(frozen.fixture_bytes(self.cases)).hexdigest()
        self.assertEqual(digest, frozen.EXPECTED_FIXTURE_SHA256)

    def test_validation_fixture_is_frozen_with_its_own_denominators(self):
        cases = frozen.load_cases(frozen.DEFAULT_PACKAGE, "validation")
        counts = {
            "queries": len(cases),
            "groups": len({case["group_id"] for case in cases}),
            "useful": sum(any(grade >= 2 for grade in case["qrels"].values()) for case in cases),
            "direct": sum(case["best_settled_label"] == "direct" for case in cases),
        }
        self.assertEqual(counts, {"queries": 260, "groups": 257, "useful": 221, "direct": 40})
        self.assertEqual(
            hashlib.sha256(frozen.fixture_bytes(cases)).hexdigest(),
            frozen.EXPECTED_VALIDATION_FIXTURE_SHA256,
        )


class RankingValidationTests(unittest.TestCase):
    def setUp(self):
        self.cases = [
            {"id": "q1", "group_id": "g1", "qrels": {"1:1": 3}},
            {"id": "q2", "group_id": "g2", "qrels": {"1:2": 2}},
        ]
        self.corpus_refs = {"1:1", "1:2", "2:1"}
        self.report = {
            "runs": {
                "q1": {"refs": ["1:1"], "ms": 1.0},
                "q2": {"refs": ["1:2"], "ms": 2.0},
            }
        }

    def test_valid_rankings(self):
        rankings = frozen.validate_run(self.cases, self.report, self.corpus_refs)
        self.assertEqual(rankings["q1"], ["1:1"])
        self.assertEqual(rankings["q2"], ["1:2"])

    def test_missing_and_extra_query_ids_are_rejected(self):
        missing = copy.deepcopy(self.report)
        del missing["runs"]["q2"]
        with self.assertRaisesRegex(ValueError, "Ranking IDs mismatch"):
            frozen.validate_run(self.cases, missing, self.corpus_refs)

        extra = copy.deepcopy(self.report)
        extra["runs"]["q3"] = {"refs": [], "ms": 1.0}
        with self.assertRaisesRegex(ValueError, "Ranking IDs mismatch"):
            frozen.validate_run(self.cases, extra, self.corpus_refs)

    def test_invalid_rankings_are_rejected(self):
        mutations = (
            (["1:1", "1:1"], "duplicate ayah"),
            (["01:1"], "invalid ayah"),
            (["9:9"], "unknown ayat"),
            ([f"2:{index}" for index in range(1, 52)], "invalid ranked result list"),
        )
        for refs, message in mutations:
            with self.subTest(message=message):
                report = copy.deepcopy(self.report)
                report["runs"]["q1"]["refs"] = refs
                with self.assertRaisesRegex(ValueError, message):
                    frozen.validate_run(self.cases, report, self.corpus_refs)

    def test_malformed_entries_and_latencies_are_rejected(self):
        report = copy.deepcopy(self.report)
        report["runs"]["q1"] = []
        with self.assertRaisesRegex(ValueError, "entry must be an object"):
            frozen.validate_run(self.cases, report, self.corpus_refs)

        for latency in (None, -1, float("nan"), "bad"):
            with self.subTest(latency=latency):
                report = copy.deepcopy(self.report)
                report["runs"]["q1"]["ms"] = latency
                with self.assertRaisesRegex(ValueError, "latency"):
                    frozen.validate_run(self.cases, report, self.corpus_refs)


class ScoringAndReportingTests(unittest.TestCase):
    def test_direct_slice_uses_useful_hit_not_direct_hit(self):
        score_group = frozen.scorer_module(frozen.DEFAULT_SCORER_PACKAGE)
        case = {
            "id": "q1",
            "group_id": "g1",
            "query": "question",
            "best_settled_label": "direct",
            "compound_status": "not_flagged",
            "qrels": {"1:1": 3, "1:2": 2, "1:3": 1},
            "disputed_ayahs": [],
        }
        scores = frozen.score_run(
            [case], {"q1": ["1:3", "1:2"]}, score_group
        )["q1"]
        self.assertEqual(scores["positive_hit@5"], 1.0)
        self.assertEqual(scores["useful_hit@5"], 1.0)
        self.assertEqual(scores["direct_hit@5"], 0.0)
        self.assertIsNone(scores["useful_recall@50"])

        refs = [f"2:{index}" for index in range(1, 50)] + ["1:2"]
        scores = frozen.score_run([case], {"q1": refs}, score_group)["q1"]
        self.assertEqual(scores["useful_recall@50"], 0.5)

    def test_rank_regressions_report_the_lost_rank(self):
        cases = [{"id": "q1", "group_id": "g1", "query": "question", "qrels": {"1:1": 3, "1:2": 1}}]
        details = frozen.rank_regression_details(
            cases,
            {"q1": ["1:1", "1:2"]},
            {"q1": ["1:2", "1:1"]},
            minimum_grade=2,
        )
        self.assertEqual(details["count"], 1)
        self.assertEqual(details["cases"][0]["baseline_rank"], 1)
        self.assertEqual(details["cases"][0]["candidate_rank"], 2)

    def test_leaderboard_keeps_distinct_baselines_and_new_column(self):
        summary = {
            "runs": {
                name: {
                    "headline": {
                        slice_name: {"metrics": {}, "counts": {}}
                        for slice_name in ("all", "useful", "direct")
                    }
                }
                for name, _ in frozen.RUN_COLUMNS
            }
        }
        table = frozen.markdown_leaderboard(summary)
        for _, label in frozen.RUN_COLUMNS:
            self.assertIn(label, table.splitlines()[0])
        self.assertIn("Useful hit@5, direct-answer queries", table)

    def test_m01_column_is_appended_without_replacing_baselines(self):
        summary = {
            "runs": {
                **{
                    name: {"headline": {slice_name: {"metrics": {}, "counts": {}} for slice_name in ("all", "useful", "direct")}}
                    for name, _ in frozen.RUN_COLUMNS
                },
                "m01": {
                    "leaderboard_label": "M01 mmr λ=0.85 k=50",
                    "headline": {slice_name: {"metrics": {}, "counts": {}} for slice_name in ("all", "useful", "direct")},
                },
            }
        }
        header = frozen.markdown_leaderboard(summary).splitlines()[0]
        self.assertIn("PR #2 browser | M00 offline E5/FAISS | M01 mmr λ=0.85 k=50", header)

    def test_m02_column_is_appended_after_m01(self):
        summary = {
            "runs": {
                **{
                    name: {"headline": {slice_name: {"metrics": {}, "counts": {}} for slice_name in ("all", "useful", "direct")}}
                    for name, _ in frozen.RUN_COLUMNS
                },
                "m01": {
                    "leaderboard_label": "M01 similarity λ=1 k=50",
                    "headline": {slice_name: {"metrics": {}, "counts": {}} for slice_name in ("all", "useful", "direct")},
                },
                "m02": {
                    "leaderboard_label": "M02 translation_ai",
                    "headline": {slice_name: {"metrics": {}, "counts": {}} for slice_name in ("all", "useful", "direct")},
                },
            }
        }
        header = frozen.markdown_leaderboard(summary).splitlines()[0]
        self.assertIn("M01 similarity λ=1 k=50 | M02 translation_ai", header)

    def test_m02_headline_label_does_not_require_m01_fields(self):
        cases = [{
            "id": "q1",
            "group_id": "g1",
            "query": "question",
            "best_settled_label": "direct",
            "compound_status": "not_flagged",
            "qrels": {"1:1": 3},
            "disputed_ayahs": [],
        }]
        report = {
            "engine": "M02",
            "config": {"semantic_ai": False, "representation_scope": "translation"},
            "field": "translation",
            "source_attribution": "per-result winning representation",
            "runs": {"q1": {"refs": ["1:1"], "ms": 1.0}},
            "index_bytes": 1,
            "peak_rss_bytes": 1,
        }
        scores = {"q1": {"positive_hit@5": 1.0, "useful_hit@5": 1.0, "ndcg@10": 1.0, "useful_recall@50": None}}
        with mock.patch.object(frozen, "aggregate", return_value={"queries": 1, "groups": 1, "metrics": {}}):
            summary = frozen.candidate_headline_summary(report, cases, {"q1": ["1:1"]}, scores, 0, 1)
        self.assertEqual(summary["leaderboard_label"], "M02 translation")

    def test_m03_column_is_appended_after_m02(self):
        summary = {
            "runs": {
                **{
                    name: {"headline": {slice_name: {"metrics": {}, "counts": {}} for slice_name in ("all", "useful", "direct")}}
                    for name, _ in frozen.RUN_COLUMNS
                },
                "m01": {
                    "leaderboard_label": "M01 similarity λ=1 k=50",
                    "headline": {slice_name: {"metrics": {}, "counts": {}} for slice_name in ("all", "useful", "direct")},
                },
                "m02": {
                    "leaderboard_label": "M02 translation",
                    "headline": {slice_name: {"metrics": {}, "counts": {}} for slice_name in ("all", "useful", "direct")},
                },
                "m03": {
                    "leaderboard_label": "M03 translation_en_tafsir",
                    "headline": {slice_name: {"metrics": {}, "counts": {}} for slice_name in ("all", "useful", "direct")},
                },
            }
        }
        header = frozen.markdown_leaderboard(summary).splitlines()[0]
        self.assertIn("M02 translation | M03 translation_en_tafsir", header)

    def test_m03_headline_label_uses_its_experiment(self):
        cases = [{
            "id": "q1", "group_id": "g1", "query": "question",
            "best_settled_label": "direct", "compound_status": "not_flagged",
            "qrels": {"1:1": 3}, "disputed_ayahs": [],
        }]
        report = {
            "engine": "M03", "experiment": "M03",
            "config": {"semantic_ai": False, "semantic_en_tafsir": True, "representation_scope": "translation_en_tafsir"},
            "field": "translation", "source_attribution": "per-result winning representation",
            "runs": {"q1": {"refs": ["1:1"], "ms": 1.0}},
            "index_bytes": 1, "peak_rss_bytes": 1,
        }
        scores = {"q1": {"positive_hit@5": 1.0, "useful_hit@5": 1.0, "ndcg@10": 1.0, "useful_recall@50": None}}
        with mock.patch.object(frozen, "aggregate", return_value={"queries": 1, "groups": 1, "metrics": {}}):
            summary = frozen.candidate_headline_summary(report, cases, {"q1": ["1:1"]}, scores, 0, 1)
        self.assertEqual(summary["leaderboard_label"], "M03 translation_en_tafsir")

    def test_validation_gate_requires_useful_gain_without_ndcg_loss(self):
        def summary(useful, ndcg):
            return {
                "headline": {
                    "useful": {"counts": {"useful_hit@5": useful}},
                    "all": {"metrics": {"ndcg@10": {"mean": ndcg}}},
                }
            }

        parent = summary(10, 0.1)
        self.assertFalse(frozen.validation_gate(parent, summary(10, 0.2)))
        self.assertFalse(frozen.validation_gate(parent, summary(11, 0.09)))
        self.assertTrue(frozen.validation_gate(parent, summary(11, 0.1)))
        self.assertIs(frozen.choose_validation_candidate(parent, [summary(10, 0.2)]), parent)

    def test_config_hash_is_stable_and_rejects_unrecorded_values(self):
        config = {"retrieval_mode": "mmr", "fetch_k": 50, "mmr_lambda": 0.85}
        self.assertEqual(frozen.config_sha256(config), frozen.config_sha256(dict(reversed(list(config.items())))))
        with self.assertRaisesRegex(ValueError, "fetch_k"):
            frozen.canonical_config({**config, "fetch_k": 40})
        with self.assertRaisesRegex(ValueError, "mmr_lambda"):
            frozen.canonical_config({**config, "mmr_lambda": 0.8})

    def test_runtime_budgets_are_fixed_from_measured_baseline(self):
        budgets = frozen.runtime_budgets(
            {
                "dense": {
                    "timings_ms": {"p50": 50, "p95": 100, "queries": 345},
                    "index_bytes": 1000,
                    "peak_rss_bytes": 2000,
                }
            }
        )
        self.assertEqual(budgets["headroom"], 1.10)
        self.assertEqual(
            budgets["runs"]["dense"],
            {
                "p95_query_ms_max": 110,
                "index_bytes_max": 1100,
                "peak_rss_bytes_max": 2200,
            },
        )


class EntrypointTests(unittest.TestCase):
    def test_frozen_profile_routes_before_legacy_imports(self):
        env = os.environ.copy()
        env["EVAL_PROFILE"] = "pr2_frozen_v1"
        result = subprocess.run(
            [sys.executable, str(frozen.ROOT / "run_benchmark.py"), "--help"],
            cwd=frozen.ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf8",
        )
        self.assertIn("Frozen IslamQA v3 evaluation contract", result.stdout)
        self.assertIn("prepare", result.stdout)
        self.assertIn("dense-sweep", result.stdout)
        self.assertIn("m01-report", result.stdout)
        self.assertIn("m02-export", result.stdout)
        self.assertIn("m02-sweep", result.stdout)
        self.assertIn("m02-report", result.stdout)


if __name__ == "__main__":
    unittest.main()
