from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT / "evaluation")):
    if path not in sys.path:
        sys.path.insert(0, path)

import m04_translation
import representations
import utils
from test_ai_representation import FakeDatabase, FakeDocument, deployment_with


class ExtraDocument(FakeDocument):
    def __init__(self, surah, ayah, text="extra translation", score_id=None):
        super().__init__(surah, ayah, "translation", score_id=score_id, text=text)
        self.metadata = {
            "surah": surah,
            "ayah": ayah,
            "field": "extra_translation",
            "source": "pickthall_1930",
        }
        self.id = score_id or f"extra_translation:{surah}:{ayah}"


def write_translation_csv(path: Path, rows: list[tuple[str, str, str]]) -> None:
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Name", "Surah", "Ayat", "Verse", "Tafseer"])
        for surah, ayah, verse in rows:
            writer.writerow(["Test", surah, ayah, verse, "ignored commentary"])


def extra_deployment(translation=(), extra=()):
    deployment = deployment_with(translation)
    deployment.extra_translation_id = "none"
    deployment.extra_translation_db = FakeDatabase(list(extra)) if extra else None
    deployment.extra_translation_db_docstore_id_to_index = (
        dict(zip(deployment.extra_translation_db.index_to_docstore_id.values(), deployment.extra_translation_db.index_to_docstore_id.keys()))
        if deployment.extra_translation_db is not None else {}
    )
    return deployment


class ExtraTranslationExportTests(unittest.TestCase):
    def test_complete_mapping_excludes_exact_current_duplicates_and_preserves_display(self):
        class FakeTokenizer:
            def encode(self, text, add_special_tokens, truncation):
                return list(range(len(text.split()) + 2))

        corpus = {
            "1:1": "Same, wording.",
            "1:2": "Retained second.",
            "1:3": "Retained third.",
            "1:4": "Retained fourth.",
        }
        original = dict(corpus)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source_csv = directory / "translation.csv"
            write_translation_csv(source_csv, [
                ("1", "1", "Same wording!"),
                ("1", "2", "A genuinely different second verse, with comma."),
                ("1", "3", "A genuinely different third verse."),
                ("1", "4", ""),
            ])
            sources = directory / "sources.json"
            sources.write_text("[]", encoding="utf8")
            output = directory / "extra.jsonl"
            digest = hashlib.sha256(source_csv.read_bytes()).hexdigest()
            with mock.patch.object(representations, "EXTRA_TRANSLATION_CSV_SHA256", digest):
                manifest = representations.export_extra_translation_representation(
                    source_csv, corpus, output, [sources], tokenizer=FakeTokenizer(), max_tokens=5
                )
            rows = [json.loads(line) for line in output.read_text(encoding="utf8").splitlines()]
        self.assertEqual(corpus, original)
        self.assertEqual([row["ayah"] for row in rows], ["1:2", "1:3"])
        self.assertTrue(all(row["field"] == "extra_translation" and row["source"] == "pickthall_1930" for row in rows))
        self.assertEqual(rows[0]["text"], "A genuinely different second verse, with comma.")
        self.assertEqual(manifest["counts"], {
            "source_rows": 4,
            "valid_text": 3,
            "exported": 2,
            "malformed_text": 0,
            "current_translation_duplicates_excluded": 1,
            "overlap_excluded": 0,
            "missing": 2,
            "empty_text": 1,
        })
        self.assertEqual(manifest["current_translation_duplicate_ayahs"], ["1:1"])
        self.assertEqual(manifest["missing_ayahs"], ["1:1", "1:4"])
        self.assertTrue(manifest["mapping"]["exact_reference_set"])
        self.assertTrue(manifest["mapping"]["canonical_order"])
        self.assertTrue(manifest["mapping"]["per_surah_counts_match"])
        self.assertEqual(manifest["display_field"], "translation")
        self.assertEqual(manifest["translation_independence"]["exact_normalized_duplicates"], 1)
        self.assertEqual(manifest["translation_independence"]["different_ayahs"], 2)
        self.assertIn("falls back", manifest["missing_verse_fallback"])

    def test_quoted_csv_fields_are_preserved(self):
        corpus = {"1:1": "retained"}
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source_csv = directory / "translation.csv"
            write_translation_csv(source_csv, [("1", "1", 'Different, "quoted" wording.')])
            sources = directory / "sources.json"
            sources.write_text("[]", encoding="utf8")
            output = directory / "extra.jsonl"
            digest = hashlib.sha256(source_csv.read_bytes()).hexdigest()
            with mock.patch.object(representations, "EXTRA_TRANSLATION_CSV_SHA256", digest):
                representations.export_extra_translation_representation(source_csv, corpus, output, [sources])
            row = json.loads(output.read_text(encoding="utf8"))
        self.assertEqual(row["text"], 'Different, "quoted" wording.')

    def test_duplicate_unknown_and_noncanonical_numbering_are_rejected(self):
        cases = (
            [("1", "1", "one"), ("1", "1", "two")],
            [("2", "1", "unknown")],
            [("01", "1", "noncanonical")],
        )
        for rows in cases:
            with self.subTest(rows=rows), tempfile.TemporaryDirectory() as directory:
                directory = Path(directory)
                source_csv = directory / "translation.csv"
                write_translation_csv(source_csv, rows)
                sources = directory / "sources.json"
                sources.write_text("[]", encoding="utf8")
                digest = hashlib.sha256(source_csv.read_bytes()).hexdigest()
                with mock.patch.object(representations, "EXTRA_TRANSLATION_CSV_SHA256", digest):
                    with self.assertRaises(ValueError):
                        representations.export_extra_translation_representation(
                            source_csv, {"1:1": "retained"}, directory / "extra.jsonl", [sources]
                        )

    def test_missing_source_rows_have_explicit_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source_csv = directory / "translation.csv"
            write_translation_csv(source_csv, [("1", "1", "different")])
            sources = directory / "sources.json"
            sources.write_text("[]", encoding="utf8")
            digest = hashlib.sha256(source_csv.read_bytes()).hexdigest()
            with mock.patch.object(representations, "EXTRA_TRANSLATION_CSV_SHA256", digest):
                manifest = representations.export_extra_translation_representation(
                    source_csv, {"1:1": "retained", "1:2": "missing"}, directory / "extra.jsonl", [sources]
                )
        self.assertFalse(manifest["mapping"]["exact_reference_set"])
        self.assertEqual(manifest["mapping"]["missing_source_ayahs"], ["1:2"])
        self.assertEqual(manifest["missing_ayahs"], ["1:2"])
        self.assertIn("extra-only scope cannot return", manifest["missing_verse_fallback"])

    def test_source_overlap_is_recorded_without_selective_exclusion(self):
        tokens = [f"token{index}" for index in range(50)]
        text = "prefix " + " ".join(tokens) + " suffix"
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source_csv = directory / "translation.csv"
            write_translation_csv(source_csv, [("1", "1", text)])
            sources = directory / "sources.json"
            sources.write_text(json.dumps([{"question": " ".join(tokens), "answer": ""}]), encoding="utf8")
            output = directory / "extra.jsonl"
            digest = hashlib.sha256(source_csv.read_bytes()).hexdigest()
            with mock.patch.object(representations, "EXTRA_TRANSLATION_CSV_SHA256", digest):
                manifest = representations.export_extra_translation_representation(
                    source_csv, {"1:1": "retained"}, output, [sources]
                )
        self.assertEqual(manifest["counts"]["exported"], 1)
        self.assertEqual(manifest["counts"]["overlap_excluded"], 0)
        self.assertEqual(manifest["provenance_audit"]["overlap_count"], 1)
        self.assertEqual(manifest["provenance_audit"]["overlap_detected_rows"], [{"ayah": "1:1", "matching_shingles": 1}])
        self.assertIn("without excluding rows", manifest["provenance_audit"]["exclusion_rule"])


class ExtraTranslationRetrievalTests(unittest.TestCase):
    def test_combined_scope_keeps_max_score_and_one_result_per_ayah(self):
        translation = [(FakeDocument(1, 1), 0.8), (FakeDocument(1, 2), 0.7)]
        extra = [(ExtraDocument(1, 1), 0.9), (ExtraDocument(1, 3), 0.85)]
        deployment = extra_deployment(translation, extra)
        results = deployment.search(
            "best_match", "query", k=3, extra_translation_id="pickthall_1930",
            representation_scope="translation_extra_translation",
        )
        self.assertEqual(
            [(doc.metadata["surah"], doc.metadata["ayah"], doc.metadata["field"], score) for doc, score in results],
            [(1, 1, "extra_translation", 0.9), (1, 3, "extra_translation", 0.85), (1, 2, "translation", 0.7)],
        )
        self.assertEqual(deployment.db.similarity_calls, [([0.25, 0.75], 50)])
        self.assertEqual(deployment.extra_translation_db.similarity_calls, [([0.25, 0.75], 50)])

    def test_equal_scores_prefer_retained_translation_then_numeric_ayah(self):
        translation = [(FakeDocument(1, 10), 0.5), (FakeDocument(1, 2), 0.5), (FakeDocument(1, 1), 0.5)]
        extra = [(ExtraDocument(1, 2), 0.5)]
        deployment = extra_deployment(translation, extra)
        results = deployment.search(
            "best_match", "query", k=3, extra_translation_id="pickthall_1930",
            representation_scope="translation_extra_translation",
        )
        self.assertEqual([doc.id for doc, _ in results], ["translation:1:1", "translation:1:2", "translation:1:10"])

    def test_extra_only_uses_only_extra_index(self):
        deployment = extra_deployment([(FakeDocument(1, 1), 0.8)], [(ExtraDocument(1, 2), 0.9)])
        results = deployment.search(
            "best_match", "query", k=1, extra_translation_id="pickthall_1930",
            representation_scope="extra_translation",
        )
        self.assertEqual([doc.metadata["field"] for doc, _ in results], ["extra_translation"])
        self.assertEqual(deployment.db.similarity_calls, [])
        self.assertEqual(deployment.extra_translation_db.similarity_calls, [([0.25, 0.75], 50)])

    def test_translation_rollback_never_queries_extra_index(self):
        deployment = extra_deployment(
            [(FakeDocument(1, 1), 0.8), (FakeDocument(1, 2), 0.7)],
            [(ExtraDocument(2, 1), 1.0)],
        )
        parent = deployment.search("best_match", "query", k=2)
        deployment.extra_translation_id = "pickthall_1930"
        rollback = deployment.search(
            "best_match", "query", k=2, extra_translation_id="pickthall_1930",
            representation_scope="translation",
        )
        self.assertEqual(parent, rollback)
        self.assertEqual(deployment.extra_translation_db.similarity_calls, [])
        self.assertTrue(all(doc.metadata["field"] == "translation" for doc, _ in rollback))

    def test_enabled_translation_id_selects_combined_default_scope(self):
        deployment = extra_deployment([(FakeDocument(1, 1), 0.5)], [(ExtraDocument(1, 2), 0.5)])
        deployment.extra_translation_id = "pickthall_1930"
        results = deployment.search("best_match", "query", k=2)
        self.assertEqual([doc.metadata["field"] for doc, _ in results], ["translation", "extra_translation"])

    def test_invalid_extra_combinations_are_rejected_before_embedding(self):
        deployment = extra_deployment([(FakeDocument(1, 1), 1.0)])
        invalid = (
            {"representation_scope": "extra_translation"},
            {"extra_translation_id": "pickthall_1930", "representation_scope": "extra_translation"},
            {"extra_translation_id": "other", "representation_scope": "translation"},
            {"extra_translation_id": "pickthall_1930", "representation_scope": "translation_extra_translation", "retrieval_mode": "mmr"},
        )
        for overrides in invalid:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    deployment.search("best_match", "query", k=1, **overrides)
        self.assertEqual(deployment.embeddings.calls, [])

    def test_display_document_uses_retained_translation_without_changing_attribution(self):
        retained = FakeDocument(1, 1, text="retained Saheeh International")
        extra = ExtraDocument(1, 1, text="Pickthall wording")
        deployment = extra_deployment([(retained, 0.8)], [(extra, 0.9)])
        deployment.translation_docs_by_ayah = {"1:1": retained}
        self.assertIs(deployment.display_document(extra), retained)
        self.assertEqual(extra.page_content, "Pickthall wording")
        self.assertEqual(extra.metadata["field"], "extra_translation")
        self.assertIs(deployment.display_document(retained), retained)

    def test_environment_switch_defaults_off_and_validates_path(self):
        translation_db = FakeDatabase([])
        extra_db = FakeDatabase([])
        with mock.patch.dict(os.environ, {"EXTRA_TRANSLATION_ID": "pickthall_1930"}, clear=False):
            with mock.patch.object(utils, "HuggingFaceEmbeddings"):
                with self.assertRaisesRegex(ValueError, "requires extra_translation_index_path"):
                    utils.VectorSearchDeployment("unused", "unused", "cpu")
        with mock.patch.dict(os.environ, {"EXTRA_TRANSLATION_ID": "invalid"}, clear=False):
            with mock.patch.object(utils, "HuggingFaceEmbeddings"):
                with self.assertRaisesRegex(ValueError, "must be 'none' or 'pickthall_1930'"):
                    utils.VectorSearchDeployment("unused", "unused", "cpu")
        with mock.patch.dict(os.environ, {"EXTRA_TRANSLATION_ID": "none"}, clear=False):
            with mock.patch.object(utils, "HuggingFaceEmbeddings"):
                with mock.patch.object(utils.FAISS, "load_local", return_value=translation_db):
                    deployment = utils.VectorSearchDeployment("unused", "unused", "cpu")
        self.assertEqual(deployment.extra_translation_id, "none")
        self.assertIsNone(deployment.extra_translation_db)
        with mock.patch.dict(os.environ, {"EXTRA_TRANSLATION_ID": "pickthall_1930"}, clear=False):
            with mock.patch.object(utils, "HuggingFaceEmbeddings"):
                with mock.patch.object(utils.FAISS, "load_local", side_effect=[translation_db, extra_db]):
                    deployment = utils.VectorSearchDeployment("unused", "unused", "cpu", extra_translation_index_path="extra")
        self.assertIs(deployment.extra_translation_db, extra_db)


class M04ContractTests(unittest.TestCase):
    def test_configuration_is_canonical_and_requires_matching_switch(self):
        config = m04_translation.CONFIGS["parent_pickthall"]
        self.assertEqual(m04_translation.config_sha256(config), m04_translation.config_sha256(dict(reversed(list(config.items())))))
        with self.assertRaisesRegex(ValueError, "require extra_translation_id=pickthall_1930"):
            m04_translation.canonical_config({**config, "extra_translation_id": "none"})
        with self.assertRaisesRegex(ValueError, "must be none or pickthall_1930"):
            m04_translation.canonical_config({**config, "extra_translation_id": "other"})
        with self.assertRaisesRegex(ValueError, "must be a boolean"):
            m04_translation.canonical_config({**config, "semantic_ai": "false"})
        self.assertEqual(m04_translation.SCOPE_FIELDS, {**utils.REPRESENTATION_SCOPES, **utils.EXTRA_TRANSLATION_SCOPES})

    def test_execution_records_winning_fields_and_deduplicates_ayahs(self):
        deployment = extra_deployment(
            [(FakeDocument(1, 1), 0.8)],
            [(ExtraDocument(1, 1), 0.9), (ExtraDocument(1, 2), 0.7)],
        )
        report = m04_translation.execute_config(
            deployment, [{"id": "q1", "query": "question"}], m04_translation.CONFIGS["parent_pickthall"]
        )
        self.assertEqual(report["runs"]["q1"]["refs"], ["1:1", "1:2"])
        self.assertEqual(report["runs"]["q1"]["source_fields"], [["extra_translation"], ["extra_translation"]])

    def test_source_attribution_rejects_leakage_and_malformed_values(self):
        for name, config in m04_translation.CONFIGS.items():
            fields = m04_translation.SCOPE_FIELDS[config["representation_scope"]]
            report = {"config": config, "runs": {"q1": {"refs": ["1:1"], "source_fields": [[fields[0]]]}}}
            self.assertTrue(m04_translation.source_attribution_valid(report), name)
        malformed = ([["translation"], ["extra_translation"]], [[["extra_translation"]]], [None], "extra_translation", None, 1)
        for source_fields in malformed:
            with self.subTest(source_fields=source_fields):
                report = {"config": m04_translation.CONFIGS["parent_pickthall"], "runs": {"q1": {"refs": ["1:1"], "source_fields": source_fields}}}
                self.assertFalse(m04_translation.source_attribution_valid(report))
        self.assertFalse(m04_translation.source_attribution_valid({"config": m04_translation.CONFIGS["parent_translation"], "runs": {"q1": []}}))

    def test_headline_summary_uses_m04_label_and_preserves_config(self):
        config = m04_translation.CONFIGS["parent_pickthall"]
        report = {"config": config, "config_sha256": m04_translation.config_sha256(config)}
        headline = {"provenance": {"selection": {"frozen": True}}}
        with mock.patch.object(m04_translation.m03_tafsir.frozen, "candidate_headline_summary", return_value=headline):
            summary = m04_translation._headline_summary(report, [], {}, {}, 0, 1)
        self.assertEqual(summary["config"], config)
        self.assertEqual(summary["leaderboard_label"], "M04 translation_extra_translation")
        self.assertNotIn("selection", summary["provenance"])

    def test_index_artifacts_are_hashed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "index.faiss").write_bytes(b"vectors")
            (path / "index.pkl").write_bytes(b"docstore")
            artifacts = m04_translation.index_artifacts(path)
        self.assertEqual(artifacts["index.faiss"]["sha256"], hashlib.sha256(b"vectors").hexdigest())
        self.assertEqual(artifacts["index.pkl"]["bytes"], 8)


if __name__ == "__main__":
    unittest.main()
