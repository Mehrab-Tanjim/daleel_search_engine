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

import m02_ai
import representations
import utils


class FakeDocument:
    def __init__(self, surah, ayah, field="translation", score_id=None, text="text"):
        self.metadata = {
            "surah": surah,
            "ayah": ayah,
            "field": field,
            "source": "saheeh_international" if field == "translation" else "aayatun_ai_generated",
        }
        self.page_content = text
        self.id = score_id or f"{field}:{surah}:{ayah}"


class FakeEmbeddings:
    def __init__(self):
        self.calls = []

    def embed_query(self, query):
        self.calls.append(query)
        return [0.25, 0.75]


class FakeIndex:
    def reconstruct(self, index):
        return [float(index), 1.0]


class FakeDatabase:
    def __init__(self, candidates):
        self.candidates = candidates
        self.similarity_calls = []
        self.mmr_calls = []
        self.index_to_docstore_id = {
            index: doc.id for index, (doc, _) in enumerate(candidates)
        }
        self.index = FakeIndex()

    def similarity_search_with_score_by_vector(self, embedding, k):
        self.similarity_calls.append((embedding, k))
        return self.candidates[:k]

    def max_marginal_relevance_search_with_score_by_vector(
        self, embedding, k, fetch_k, lambda_mult
    ):
        self.mmr_calls.append((embedding, k, fetch_k, lambda_mult))
        return self.candidates[:k]


def deployment_with(translation=(), ai=()):
    deployment = object.__new__(utils.VectorSearchDeployment)
    deployment.retrieval_mode = "similarity"
    deployment.fetch_k = 50
    deployment.mmr_lambda = 1.0
    deployment.semantic_ai = False
    deployment.embeddings = FakeEmbeddings()
    deployment.db = FakeDatabase(list(translation))
    deployment.ai_db = FakeDatabase(list(ai)) if ai else None
    deployment.db_docstore_id_to_index = dict(
        zip(deployment.db.index_to_docstore_id.values(), deployment.db.index_to_docstore_id.keys())
    )
    deployment.ai_db_docstore_id_to_index = (
        dict(zip(deployment.ai_db.index_to_docstore_id.values(), deployment.ai_db.index_to_docstore_id.keys()))
        if deployment.ai_db is not None
        else {}
    )
    return deployment


class RepresentationExportTests(unittest.TestCase):
    def test_flattening_matches_aayatun_fields_and_normalization(self):
        raw = json.dumps({
            "verse_context": "  Verse context https://example.test/path  ",
            "verse_context_continuation": "must not be indexed",
            "word_meanings": [
                {
                    "word": "word",
                    "classical_meaning": "classical",
                    "meaning_here": "here",
                    "notable_usage": "usage",
                },
                "ignored non-object",
            ],
        })
        self.assertEqual(
            representations.flatten_ai_content(raw),
            "Verse context word classical here usage",
        )

    def test_malformed_or_non_object_json_is_not_exported(self):
        for raw in ("{bad", "[]", "null"):
            with self.subTest(raw=raw):
                self.assertEqual(representations.flatten_ai_content(raw), "")

    def test_exact_long_span_overlap_excludes_the_whole_ayah(self):
        tokens = [f"token{index}" for index in range(12)]
        rows = [{"ayah": "1:1", "text": "prefix " + " ".join(tokens) + " suffix"}]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "sources.json"
            source.write_text(json.dumps([{"question": " ".join(tokens), "answer": ""}]), encoding="utf8")
            audit = representations.audit_source_overlap(rows, [source])
        self.assertEqual(audit["included_rows"], [])
        self.assertEqual(audit["excluded_rows"], [{"ayah": "1:1", "matching_shingles": 1}])
        self.assertEqual(audit["overlap_count"], 1)
        self.assertEqual(audit["overlaps"][0]["field"], "question")

    def test_export_records_zero_based_conversion_missing_empty_and_truncation(self):
        class FakeTokenizer:
            def encode(self, text, add_special_tokens, truncation):
                self.asserted = (add_special_tokens, truncation)
                return list(range(len(text.split()) + 2))

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            ai_db = directory / "ai.db"
            with closing(sqlite3.connect(ai_db)) as db:
                db.execute("CREATE TABLE verse_analysis (surah INTEGER, verse INTEGER, content TEXT, PRIMARY KEY (surah, verse))")
                db.executemany(
                    "INSERT INTO verse_analysis VALUES (?, ?, ?)",
                    [
                        (1, 0, json.dumps({"verse_context": "one two three four five six seven eight"})),
                        (1, 1, "{bad"),
                        (1, 2, json.dumps({"verse_context": ""})),
                    ],
                )
                db.commit()
            source = directory / "sources.json"
            source.write_text("[]", encoding="utf8")
            output = directory / "ai_docs.jsonl"
            digest = hashlib.sha256(ai_db.read_bytes()).hexdigest()
            with mock.patch.object(representations, "AI_DB_SHA256", digest):
                manifest = representations.export_ai_representation(
                    ai_db,
                    {"1:1": "first", "1:2": "second", "1:3": "third", "1:4": "fourth"},
                    output,
                    [source],
                    tokenizer=FakeTokenizer(),
                    max_tokens=5,
                )
            rows = [json.loads(line) for line in output.read_text(encoding="utf8").splitlines()]
            output_digest = hashlib.sha256(output.read_bytes()).hexdigest()
        self.assertEqual([row["ayah"] for row in rows], ["1:1"])
        self.assertEqual(manifest["counts"], {
            "source_rows": 3,
            "valid_json": 2,
            "exported": 1,
            "malformed_json": 1,
            "overlap_excluded": 0,
            "missing": 3,
            "empty_text": 1,
        })
        self.assertEqual(manifest["malformed_json_ayahs"], ["1:2"])
        self.assertEqual(manifest["empty_text_ayahs"], ["1:3"])
        self.assertEqual(manifest["missing_ayahs"], ["1:2", "1:3", "1:4"])
        self.assertEqual(manifest["token_truncation"]["truncated_ayahs_file"], "truncated_ayahs.json")
        self.assertEqual(
            manifest["token_truncation"]["truncated_ayahs_sha256"],
            hashlib.sha256(b'["1:1"]').hexdigest(),
        )
        self.assertEqual(manifest["output"]["sha256"], output_digest)


class AIRetrievalTests(unittest.TestCase):
    def test_parent_ai_keeps_max_score_and_one_result_per_ayah(self):
        translation = [(FakeDocument(1, 1), 0.8), (FakeDocument(1, 2), 0.7)]
        ai = [(FakeDocument(1, 1, "ai"), 0.9), (FakeDocument(1, 3, "ai"), 0.85)]
        deployment = deployment_with(translation, ai)
        results = deployment.search(
            "best_match", "query", k=3, semantic_ai=True, representation_scope="translation_ai"
        )
        self.assertEqual(
            [(doc.metadata["surah"], doc.metadata["ayah"], doc.metadata["field"], score) for doc, score in results],
            [(1, 1, "ai", 0.9), (1, 3, "ai", 0.85), (1, 2, "translation", 0.7)],
        )
        self.assertEqual(deployment.db.similarity_calls, [([0.25, 0.75], 50)])
        self.assertEqual(deployment.ai_db.similarity_calls, [([0.25, 0.75], 50)])

    def test_equal_scores_prefer_translation_then_numeric_ayah(self):
        translation = [(FakeDocument(1, 10), 0.5), (FakeDocument(1, 2), 0.5), (FakeDocument(1, 1), 0.5)]
        ai = [(FakeDocument(1, 2, "ai"), 0.5)]
        deployment = deployment_with(translation, ai)
        results = deployment.search(
            "best_match", "query", k=3, semantic_ai=True, representation_scope="translation_ai"
        )
        self.assertEqual([doc.id for doc, _ in results], ["translation:1:1", "translation:1:2", "translation:1:10"])

    def test_translation_only_rollback_never_queries_ai(self):
        translation = [(FakeDocument(1, index), 1.0 - index / 100) for index in range(1, 6)]
        deployment = deployment_with(translation, [(FakeDocument(2, 1, "ai"), 1.0)])
        parent = deployment.search("best_match", "query", k=5)
        rollback = deployment.search(
            "best_match", "query", k=5, semantic_ai=False, representation_scope="translation"
        )
        self.assertEqual(parent, rollback)
        self.assertEqual(deployment.ai_db.similarity_calls, [])
        self.assertTrue(all(doc.metadata["field"] == "translation" for doc, _ in rollback))

    def test_ai_only_uses_only_the_ai_representation(self):
        deployment = deployment_with([(FakeDocument(1, 1), 1.0)], [(FakeDocument(2, 1, "ai"), 0.9)])
        results = deployment.search("best_match", "query", k=1, semantic_ai=True, representation_scope="ai")
        self.assertEqual([doc.metadata["field"] for doc, _ in results], ["ai"])
        self.assertEqual(deployment.db.similarity_calls, [])
        self.assertEqual(deployment.ai_db.similarity_calls, [([0.25, 0.75], 50)])

    def test_invalid_ai_combinations_are_rejected_before_embedding(self):
        deployment = deployment_with([(FakeDocument(1, 1), 1.0)])
        invalid = (
            {"representation_scope": "ai"},
            {"semantic_ai": True, "representation_scope": "ai"},
            {"semantic_ai": True, "representation_scope": "translation_ai", "retrieval_mode": "mmr"},
            {"semantic_ai": "yes"},
            {"semantic_ai": True, "representation_scope": "unknown"},
        )
        for overrides in invalid:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    deployment.search("best_match", "query", k=1, **overrides)
        self.assertEqual(deployment.embeddings.calls, [])

    def test_environment_switch_defaults_off_and_requires_an_ai_path(self):
        database = FakeDatabase([])
        with mock.patch.dict(os.environ, {"SEMANTIC_AI": "true"}, clear=False):
            with mock.patch.object(utils, "HuggingFaceEmbeddings"):
                with self.assertRaisesRegex(ValueError, "requires ai_index_path"):
                    utils.VectorSearchDeployment("unused", "unused", "cpu")
        with mock.patch.dict(os.environ, {"SEMANTIC_AI": "invalid"}, clear=False):
            with mock.patch.object(utils, "HuggingFaceEmbeddings"):
                with self.assertRaisesRegex(ValueError, "must be 'false' or 'true'"):
                    utils.VectorSearchDeployment("unused", "unused", "cpu")
        with mock.patch.dict(os.environ, {"SEMANTIC_AI": "false"}, clear=False):
            with mock.patch.object(utils, "HuggingFaceEmbeddings"):
                with mock.patch.object(utils.FAISS, "load_local", return_value=database):
                    deployment = utils.VectorSearchDeployment("unused", "unused", "cpu")
        self.assertFalse(deployment.semantic_ai)
        self.assertIsNone(deployment.ai_db)

    def test_return_embeddings_uses_the_winning_representation(self):
        translation = [(FakeDocument(1, 1), 0.8)]
        ai = [(FakeDocument(1, 1, "ai"), 0.9)]
        deployment = deployment_with(translation, ai)
        results, embeddings = deployment.search(
            "best_match",
            "query",
            k=1,
            return_embeddings=True,
            semantic_ai=True,
            representation_scope="translation_ai",
        )
        self.assertEqual(results[0][0].metadata["field"], "ai")
        self.assertEqual(embeddings, [[0.0, 1.0]])


class M02ContractTests(unittest.TestCase):
    def test_configuration_is_canonical_and_ai_requires_the_switch(self):
        config = {"semantic_ai": True, "representation_scope": "translation_ai"}
        self.assertEqual(m02_ai.config_sha256(config), m02_ai.config_sha256(dict(reversed(list(config.items())))))
        with self.assertRaisesRegex(ValueError, "AI scopes require semantic_ai=true"):
            m02_ai.canonical_config({"semantic_ai": False, "representation_scope": "ai"})
        with self.assertRaisesRegex(ValueError, "must be a boolean"):
            m02_ai.canonical_config({"semantic_ai": "true", "representation_scope": "translation_ai"})

    def test_execution_records_per_result_source_and_deduplicates_ayahs(self):
        deployment = deployment_with(
            [(FakeDocument(1, 1), 0.8)],
            [(FakeDocument(1, 1, "ai"), 0.9), (FakeDocument(1, 2, "ai"), 0.7)],
        )
        cases = [{"id": "q1", "query": "question"}]
        report = m02_ai.execute_config(deployment, cases, m02_ai.CONFIGS["parent_ai"])
        self.assertEqual(report["runs"]["q1"]["refs"], ["1:1", "1:2"])
        self.assertEqual(report["runs"]["q1"]["source_fields"], [["ai"], ["ai"]])
        self.assertEqual(deployment.embeddings.calls, ["query: question"])

    def test_source_attribution_accepts_each_representation_scope(self):
        examples = {
            "translation": [["translation"]],
            "ai": [["ai"]],
            "translation_ai": [["translation"], ["ai"]],
        }
        for scope, source_fields in examples.items():
            with self.subTest(scope=scope):
                report = {
                    "config": m02_ai.CONFIGS[
                        "parent_translation" if scope == "translation" else
                        "ai_only_diagnostic" if scope == "ai" else "parent_ai"
                    ],
                    "runs": {"q1": {"refs": [f"1:{index}" for index in range(1, len(source_fields) + 1)], "source_fields": source_fields}},
                }
                self.assertTrue(m02_ai.source_attribution_valid(report))

    def test_source_attribution_rejects_leakage_and_length_mismatch(self):
        for source_fields in ([["ai"]], [["translation"], ["ai"]]):
            with self.subTest(source_fields=source_fields):
                report = {
                    "config": m02_ai.CONFIGS["parent_translation"],
                    "runs": {"q1": {"refs": ["1:1"], "source_fields": source_fields}},
                }
                self.assertFalse(m02_ai.source_attribution_valid(report))

    def test_source_attribution_rejects_malformed_values_without_raising(self):
        malformed = ([["translation"], ["ai"]], [[["translation"]]], [None], "translation", None, 1)
        for source_fields in malformed:
            with self.subTest(source_fields=source_fields):
                report = {
                    "config": m02_ai.CONFIGS["parent_translation"],
                    "runs": {"q1": {"refs": ["1:1"], "source_fields": source_fields}},
                }
                self.assertFalse(m02_ai.source_attribution_valid(report))

    def test_index_artifacts_are_hashed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "index.faiss").write_bytes(b"vectors")
            (path / "index.pkl").write_bytes(b"docstore")
            artifacts = m02_ai.index_artifacts(path)
        self.assertEqual(artifacts["index.faiss"]["sha256"], hashlib.sha256(b"vectors").hexdigest())
        self.assertEqual(artifacts["index.pkl"]["bytes"], 8)


if __name__ == "__main__":
    unittest.main()
