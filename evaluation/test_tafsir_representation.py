from __future__ import annotations

from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT / "evaluation")):
    if path not in sys.path:
        sys.path.insert(0, path)

import m03_tafsir
import representations
import utils
from test_ai_representation import FakeDatabase, FakeDocument, deployment_with


class TafsirExportTests(unittest.TestCase):
    def test_flattening_matches_aayatun_markup_and_entities(self):
        self.assertEqual(
            representations.flatten_tafsir_content(" <p>One &amp; two</p>\n<b>three</b> "),
            "One & two three",
        )
        self.assertEqual(representations.flatten_tafsir_content(None), "")
        self.assertEqual(representations.flatten_tafsir_content("<p>&nbsp;</p>"), "")

    def test_export_preserves_ids_empty_entries_duplicates_and_truncation(self):
        class FakeTokenizer:
            def encode(self, text, add_special_tokens, truncation):
                self.asserted = (add_special_tokens, truncation)
                return list(range(len(text.split()) + 2))

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            tafsir_db = directory / "tafsir.db"
            with closing(sqlite3.connect(tafsir_db)) as db:
                db.execute("CREATE TABLE tafsir_kathir (surah INTEGER, ayah INTEGER, tafsir_text)")
                db.executemany(
                    "INSERT INTO tafsir_kathir VALUES (?, ?, ?)",
                    [
                        (1, 1, "<p>Same &amp; repeated commentary</p>"),
                        (1, 2, "Same &amp; repeated commentary"),
                        (1, 3, "<p></p>"),
                    ],
                )
                db.commit()
            source = directory / "sources.json"
            source.write_text("[]", encoding="utf8")
            output = directory / "en_tafsir_docs.jsonl"
            digest = hashlib.sha256(tafsir_db.read_bytes()).hexdigest()
            with mock.patch.object(representations, "TAFSIR_DB_SHA256", digest):
                manifest = representations.export_tafsir_representation(
                    tafsir_db,
                    {"1:1": "first", "1:2": "second", "1:3": "third", "1:4": "fourth"},
                    output,
                    [source],
                    tokenizer=FakeTokenizer(),
                    max_tokens=5,
                )
            rows = [json.loads(line) for line in output.read_text(encoding="utf8").splitlines()]
            output_digest = hashlib.sha256(output.read_bytes()).hexdigest()
        self.assertEqual([row["ayah"] for row in rows], ["1:1", "1:2"])
        self.assertEqual(set(rows[0]), {"ayah", "text", "field", "source", "source_db_sha256", "source_commit"})
        self.assertTrue(all(row["field"] == "en_tafsir" and row["source"] == "ibn_kathir_english" for row in rows))
        self.assertEqual(manifest["counts"], {
            "source_rows": 3,
            "valid_text": 3,
            "exported": 2,
            "malformed_text": 0,
            "overlap_excluded": 0,
            "missing": 2,
            "empty_text": 1,
        })
        self.assertEqual(manifest["empty_text_ayahs"], ["1:3"])
        self.assertEqual(manifest["missing_ayahs"], ["1:3", "1:4"])
        self.assertEqual(manifest["repeated_commentary"]["exported"]["exact_duplicate_text_groups"], 1)
        self.assertEqual(manifest["token_truncation"]["truncated_documents"], 2)
        self.assertEqual(manifest["output"]["sha256"], output_digest)

    def test_exact_source_overlap_excludes_the_whole_tafsir_row(self):
        tokens = [f"token{index}" for index in range(50)]
        rows = [{"ayah": "1:1", "text": "prefix " + " ".join(tokens) + " suffix"}]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "sources.json"
            source.write_text(json.dumps([{"question": " ".join(tokens), "answer": ""}]), encoding="utf8")
            audit = representations.audit_source_overlap(rows, [source], shingle_size=50, representation_field="en_tafsir")
        self.assertEqual(audit["included_rows"], [])
        self.assertEqual(audit["excluded_rows"], [{"ayah": "1:1", "matching_shingles": 1}])
        self.assertIn("en_tafsir", audit["exclusion_rule"])

    def test_repeated_fifty_token_spans_are_recorded_without_extra_votes(self):
        block = " ".join(f"token{index}" for index in range(50))
        rows = [
            {"ayah": "1:1", "text": "prefix " + block},
            {"ayah": "1:2", "text": block + " suffix"},
            {"ayah": "1:3", "text": "short"},
        ]
        summary, sidecar = representations.repeated_block_audit(rows)
        self.assertEqual(summary["documents_with_repeated_spans"], 2)
        self.assertEqual(summary["unique_repeated_spans"], 1)
        self.assertEqual(sidecar[0]["ayahs"], ["1:1", "1:2"])
        self.assertIn("do not add votes", summary["voting_control"])

    def test_duplicate_or_unknown_source_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            tafsir_db = directory / "tafsir.db"
            with closing(sqlite3.connect(tafsir_db)) as db:
                db.execute("CREATE TABLE tafsir_kathir (surah INTEGER, ayah INTEGER, tafsir_text)")
                db.executemany("INSERT INTO tafsir_kathir VALUES (?, ?, ?)", [(1, 1, "one"), (1, 1, "two")])
                db.commit()
            source = directory / "sources.json"
            source.write_text("[]", encoding="utf8")
            digest = hashlib.sha256(tafsir_db.read_bytes()).hexdigest()
            with mock.patch.object(representations, "TAFSIR_DB_SHA256", digest):
                with self.assertRaisesRegex(ValueError, "duplicate ayah"):
                    representations.export_tafsir_representation(tafsir_db, {"1:1": "one"}, directory / "out.jsonl", [source])


class TafsirRetrievalTests(unittest.TestCase):
    def test_parent_tafsir_keeps_max_score_and_one_result_per_ayah(self):
        translation = [(FakeDocument(1, 1), 0.8), (FakeDocument(1, 2), 0.7)]
        tafsir = [(FakeDocument(1, 1, "en_tafsir"), 0.9), (FakeDocument(1, 3, "en_tafsir"), 0.85)]
        deployment = deployment_with(translation, en_tafsir=tafsir)
        results = deployment.search(
            "best_match", "query", k=3, semantic_en_tafsir=True,
            representation_scope="translation_en_tafsir",
        )
        self.assertEqual(
            [(doc.metadata["surah"], doc.metadata["ayah"], doc.metadata["field"], score) for doc, score in results],
            [(1, 1, "en_tafsir", 0.9), (1, 3, "en_tafsir", 0.85), (1, 2, "translation", 0.7)],
        )
        self.assertEqual(deployment.db.similarity_calls, [([0.25, 0.75], 50)])
        self.assertEqual(deployment.en_tafsir_db.similarity_calls, [([0.25, 0.75], 50)])

    def test_equal_scores_prefer_translation_then_numeric_ayah(self):
        translation = [(FakeDocument(1, 10), 0.5), (FakeDocument(1, 2), 0.5), (FakeDocument(1, 1), 0.5)]
        tafsir = [(FakeDocument(1, 2, "en_tafsir"), 0.5)]
        deployment = deployment_with(translation, en_tafsir=tafsir)
        results = deployment.search(
            "best_match", "query", k=3, semantic_en_tafsir=True,
            representation_scope="translation_en_tafsir",
        )
        self.assertEqual([doc.id for doc, _ in results], ["translation:1:1", "translation:1:2", "translation:1:10"])

    def test_tafsir_only_isolated_and_empty_source_falls_back_to_translation(self):
        deployment = deployment_with(
            [(FakeDocument(1, 1), 0.8)],
            en_tafsir=[(FakeDocument(1, 2, "en_tafsir"), 0.9)],
        )
        only = deployment.search("best_match", "query", k=1, semantic_en_tafsir=True, representation_scope="en_tafsir")
        self.assertEqual([doc.metadata["field"] for doc, _ in only], ["en_tafsir"])
        self.assertEqual(deployment.db.similarity_calls, [])
        combined = deployment.search(
            "best_match", "query", k=2, semantic_en_tafsir=True,
            representation_scope="translation_en_tafsir",
        )
        self.assertEqual([doc.metadata["field"] for doc, _ in combined], ["en_tafsir", "translation"])
        self.assertEqual(combined[1][0].metadata["ayah"], 1)

    def test_translation_rollback_never_queries_optional_indexes(self):
        deployment = deployment_with(
            [(FakeDocument(1, index), 1.0 - index / 100) for index in range(1, 6)],
            ai=[(FakeDocument(2, 1, "ai"), 1.0)],
            en_tafsir=[(FakeDocument(3, 1, "en_tafsir"), 1.0)],
        )
        parent = deployment.search("best_match", "query", k=5)
        deployment.semantic_ai = True
        deployment.semantic_en_tafsir = True
        rollback = deployment.search(
            "best_match", "query", k=5, semantic_ai=False, semantic_en_tafsir=False,
            representation_scope="translation",
        )
        self.assertEqual(parent, rollback)
        self.assertEqual(deployment.ai_db.similarity_calls, [])
        self.assertEqual(deployment.en_tafsir_db.similarity_calls, [])
        self.assertTrue(all(doc.metadata["field"] == "translation" for doc, _ in rollback))

    def test_joint_scope_uses_all_fields_and_reconstructs_winning_embedding(self):
        deployment = deployment_with(
            [(FakeDocument(1, 1), 0.8)],
            ai=[(FakeDocument(1, 2, "ai"), 0.85)],
            en_tafsir=[(FakeDocument(1, 3, "en_tafsir"), 0.9)],
        )
        results, embeddings = deployment.search(
            "best_match", "query", k=3, return_embeddings=True,
            semantic_ai=True, semantic_en_tafsir=True,
            representation_scope="translation_ai_en_tafsir",
        )
        self.assertEqual([doc.metadata["field"] for doc, _ in results], ["en_tafsir", "ai", "translation"])
        self.assertEqual(embeddings, [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
        self.assertEqual(len(deployment.embeddings.calls), 1)

    def test_enabled_environment_flags_select_the_joint_default_scope(self):
        deployment = deployment_with(
            [(FakeDocument(1, 1), 0.5)],
            ai=[(FakeDocument(1, 2, "ai"), 0.5)],
            en_tafsir=[(FakeDocument(1, 3, "en_tafsir"), 0.5)],
        )
        deployment.semantic_ai = True
        deployment.semantic_en_tafsir = True
        results = deployment.search("best_match", "query", k=3)
        self.assertEqual([doc.metadata["field"] for doc, _ in results], ["translation", "ai", "en_tafsir"])

    def test_invalid_tafsir_combinations_are_rejected_before_embedding(self):
        deployment = deployment_with([(FakeDocument(1, 1), 1.0)])
        invalid = (
            {"representation_scope": "en_tafsir"},
            {"semantic_en_tafsir": True, "representation_scope": "en_tafsir"},
            {"semantic_en_tafsir": True, "representation_scope": "translation_en_tafsir", "retrieval_mode": "mmr"},
            {"semantic_en_tafsir": "yes"},
            {"semantic_en_tafsir": True, "representation_scope": "unknown"},
            {"semantic_ai": True, "semantic_en_tafsir": True, "representation_scope": "translation_ai_en_tafsir"},
        )
        for overrides in invalid:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    deployment.search("best_match", "query", k=1, **overrides)
        self.assertEqual(deployment.embeddings.calls, [])

    def test_environment_switch_defaults_off_and_validates_path(self):
        database = FakeDatabase([])
        with mock.patch.dict(os.environ, {"SEMANTIC_EN_TAFSIR": "true"}, clear=False):
            with mock.patch.object(utils, "HuggingFaceEmbeddings"):
                with self.assertRaisesRegex(ValueError, "requires en_tafsir_index_path"):
                    utils.VectorSearchDeployment("unused", "unused", "cpu")
        with mock.patch.dict(os.environ, {"SEMANTIC_EN_TAFSIR": "invalid"}, clear=False):
            with mock.patch.object(utils, "HuggingFaceEmbeddings"):
                with self.assertRaisesRegex(ValueError, "must be 'false' or 'true'"):
                    utils.VectorSearchDeployment("unused", "unused", "cpu")
        with mock.patch.dict(os.environ, {"SEMANTIC_EN_TAFSIR": "false"}, clear=False):
            with mock.patch.object(utils, "HuggingFaceEmbeddings"):
                with mock.patch.object(utils.FAISS, "load_local", return_value=database):
                    deployment = utils.VectorSearchDeployment("unused", "unused", "cpu")
        self.assertFalse(deployment.semantic_en_tafsir)
        self.assertIsNone(deployment.en_tafsir_db)


class M03ContractTests(unittest.TestCase):
    def test_configuration_is_canonical_and_requires_matching_switches(self):
        config = {"semantic_ai": False, "semantic_en_tafsir": True, "representation_scope": "translation_en_tafsir"}
        self.assertEqual(m03_tafsir.config_sha256(config), m03_tafsir.config_sha256(dict(reversed(list(config.items())))))
        with self.assertRaisesRegex(ValueError, "require semantic_en_tafsir=true"):
            m03_tafsir.canonical_config({**config, "semantic_en_tafsir": False})
        with self.assertRaisesRegex(ValueError, "must be a boolean"):
            m03_tafsir.canonical_config({**config, "semantic_en_tafsir": "true"})
        self.assertEqual(utils.REPRESENTATION_SCOPES, m03_tafsir.SCOPE_FIELDS)

    def test_execution_records_winning_fields_and_deduplicates_ayahs(self):
        deployment = deployment_with(
            [(FakeDocument(1, 1), 0.8)],
            en_tafsir=[(FakeDocument(1, 1, "en_tafsir"), 0.9), (FakeDocument(1, 2, "en_tafsir"), 0.7)],
        )
        report = m03_tafsir.execute_config(deployment, [{"id": "q1", "query": "question"}], m03_tafsir.CONFIGS["parent_tafsir"])
        self.assertEqual(report["runs"]["q1"]["refs"], ["1:1", "1:2"])
        self.assertEqual(report["runs"]["q1"]["source_fields"], [["en_tafsir"], ["en_tafsir"]])

    def test_source_attribution_accepts_all_scopes_and_rejects_malformed_values(self):
        for name, config in m03_tafsir.CONFIGS.items():
            fields = m03_tafsir.SCOPE_FIELDS[config["representation_scope"]]
            report = {"config": config, "runs": {"q1": {"refs": ["1:1"], "source_fields": [[fields[0]]]}}}
            self.assertTrue(m03_tafsir.source_attribution_valid(report), name)
        malformed = ([["translation"], ["en_tafsir"]], [[["en_tafsir"]]], [None], "en_tafsir", None, 1)
        for source_fields in malformed:
            with self.subTest(source_fields=source_fields):
                report = {"config": m03_tafsir.CONFIGS["parent_translation"], "runs": {"q1": {"refs": ["1:1"], "source_fields": source_fields}}}
                self.assertFalse(m03_tafsir.source_attribution_valid(report))
        self.assertFalse(m03_tafsir.source_attribution_valid({"config": m03_tafsir.CONFIGS["parent_translation"], "runs": {"q1": []}}))

    def test_joint_interaction_is_difference_in_differences(self):
        cases = [{"id": "q1", "group_id": "g1", "query": "question", "qrels": {"1:1": 3}}]
        parent = {"q1": {"useful_hit@5": 0.0}}
        tafsir = {"q1": {"useful_hit@5": 1.0}}
        ai = {"q1": {"useful_hit@5": 0.0}}
        joint = {"q1": {"useful_hit@5": 0.0}}
        result = m03_tafsir._interaction_details(cases, parent, tafsir, ai, joint, "useful_hit@5", 0, 1)
        self.assertEqual(result["difference_in_differences"], -1.0)
        self.assertEqual((result["wins"], result["losses"], result["ties"]), (0, 1, 0))

    def test_headline_summary_preserves_config_for_selection(self):
        config = m03_tafsir.CONFIGS["parent_tafsir"]
        report = {"config": config, "config_sha256": m03_tafsir.config_sha256(config)}
        headline = {"provenance": {"selection": {"frozen": True}}}
        with mock.patch.object(m03_tafsir.frozen, "candidate_headline_summary", return_value=headline):
            summary = m03_tafsir._headline_summary(report, [], {}, {}, 0, 1)
        self.assertEqual(summary["config"], config)
        self.assertEqual(summary["config_sha256"], report["config_sha256"])
        self.assertNotIn("selection", summary["provenance"])

    def test_index_artifacts_are_hashed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "index.faiss").write_bytes(b"vectors")
            (path / "index.pkl").write_bytes(b"docstore")
            artifacts = m03_tafsir.index_artifacts(path)
        self.assertEqual(artifacts["index.faiss"]["sha256"], hashlib.sha256(b"vectors").hexdigest())
        self.assertEqual(artifacts["index.pkl"]["bytes"], 8)


if __name__ == "__main__":
    unittest.main()
