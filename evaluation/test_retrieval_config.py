from __future__ import annotations

import math
import os
import sys
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import utils


class FakeDocument:
    def __init__(self, surah, ayah, text="similar wording", doc_id=None):
        self.metadata = {"surah": surah, "ayah": ayah}
        self.page_content = text
        self.id = doc_id or f"{surah}:{ayah}"


class FakeEmbeddings:
    def __init__(self):
        self.calls = []

    def embed_query(self, query):
        self.calls.append(query)
        return [0.25, 0.75]


class FakeDatabase:
    def __init__(self, candidates):
        self.candidates = candidates
        self.similarity_calls = []
        self.mmr_calls = []
        self.index_to_docstore_id = {
            index: doc.id for index, (doc, _) in enumerate(candidates)
        }

    def similarity_search_with_score_by_vector(self, embedding, k):
        self.similarity_calls.append((embedding, k))
        return self.candidates[:k]

    def max_marginal_relevance_search_with_score_by_vector(
        self, embedding, k, fetch_k, lambda_mult
    ):
        self.mmr_calls.append((embedding, k, fetch_k, lambda_mult))
        return self.candidates[:k]


def deployment_with(candidates):
    deployment = object.__new__(utils.VectorSearchDeployment)
    deployment.retrieval_mode = "similarity"
    deployment.fetch_k = 50
    deployment.mmr_lambda = 1.0
    deployment.embeddings = FakeEmbeddings()
    deployment.semantic_ai = False
    deployment.semantic_en_tafsir = False
    deployment.db = FakeDatabase(candidates)
    deployment.ai_db = None
    deployment.ai_db_docstore_id_to_index = {}
    deployment.en_tafsir_db = None
    deployment.en_tafsir_db_docstore_id_to_index = {}
    deployment.db_docstore_id_to_index = dict(
        zip(deployment.db.index_to_docstore_id.values(), deployment.db.index_to_docstore_id.keys())
    )
    return deployment


class RetrievalConfigurationTests(unittest.TestCase):
    def test_environment_configuration_is_exposed(self):
        database = FakeDatabase([])
        environment = {
            "RETRIEVAL_MODE": "mmr",
            "FETCH_K": "100",
            "MMR_LAMBDA": "0.85",
        }
        with mock.patch.dict(os.environ, environment, clear=False):
            with mock.patch.object(utils, "HuggingFaceEmbeddings"):
                with mock.patch.object(utils.FAISS, "load_local", return_value=database):
                    deployment = utils.VectorSearchDeployment("unused", "unused", "cpu")
        self.assertEqual(deployment.retrieval_mode, "mmr")
        self.assertEqual(deployment.fetch_k, 100)
        self.assertEqual(deployment.mmr_lambda, 0.85)

    def test_invalid_environment_configuration_is_rejected(self):
        invalid = (
            {"RETRIEVAL_MODE": "other"},
            {"FETCH_K": "0"},
            {"FETCH_K": "20.5"},
            {"MMR_LAMBDA": "nan"},
            {"MMR_LAMBDA": "1.1"},
        )
        for overrides in invalid:
            with self.subTest(overrides=overrides):
                environment = os.environ.copy()
                environment.update(overrides)
                with mock.patch.dict(os.environ, environment, clear=True):
                    with mock.patch.object(utils, "HuggingFaceEmbeddings"):
                        with mock.patch.object(utils.FAISS, "load_local", return_value=FakeDatabase([])):
                            with self.assertRaises(ValueError):
                                utils.VectorSearchDeployment("unused", "unused", "cpu")

    def test_explicit_configuration_overrides_environment_defaults(self):
        candidates = [(FakeDocument(1, 1), 0.9), (FakeDocument(1, 2), 0.8)]
        deployment = deployment_with(candidates)
        deployment.retrieval_mode = "mmr"
        deployment.fetch_k = 20
        deployment.mmr_lambda = 0.7
        results = deployment.search(
            "best_match",
            "query",
            k=2,
            retrieval_mode="similarity",
            fetch_k=100,
            mmr_lambda=1.0,
        )
        self.assertEqual([doc.metadata for doc, _ in results], [{"surah": 1, "ayah": 1}, {"surah": 1, "ayah": 2}])
        self.assertEqual(deployment.db.similarity_calls, [([0.25, 0.75], 100)])
        self.assertEqual(deployment.db.mmr_calls, [])


class RetrievalBehaviorTests(unittest.TestCase):
    def test_similarity_deduplicates_only_by_ayah_id(self):
        candidates = [
            (FakeDocument(1, 1, text="identical", doc_id="chunk-a"), 0.9),
            (FakeDocument(1, 1, text="identical", doc_id="chunk-b"), 0.89),
            (FakeDocument(1, 2, text="identical", doc_id="chunk-c"), 0.88),
        ]
        deployment = deployment_with(candidates)
        results = deployment.search("best_match", "query", k=2, fetch_k=20)
        self.assertEqual([doc.id for doc, _ in results], ["chunk-a", "chunk-c"])
        self.assertEqual(deployment.db.similarity_calls[0][1], 20)

    def test_legacy_dedup_retains_distinct_similar_wording_ayat(self):
        candidates = [
            (FakeDocument(1, 1, text="same wording"), 0.9),
            (FakeDocument(1, 2, text="same wording"), 0.8),
        ]
        deployment = deployment_with(candidates)
        results = deployment.relevance_search_dedup(
            "query", k=2, similarity_threshold=0.999999
        )
        self.assertEqual([doc.id for doc, _ in results], ["1:1", "1:2"])

    def test_mmr_uses_configured_depth_and_lambda(self):
        candidates = [
            (FakeDocument(1, 1), 0.9),
            (FakeDocument(1, 1), 0.85),
            (FakeDocument(1, 2), 0.8),
        ]
        deployment = deployment_with(candidates)
        results = deployment.search(
            "best_match",
            "query",
            k=2,
            retrieval_mode="mmr",
            fetch_k=100,
            mmr_lambda=0.85,
        )
        self.assertEqual([doc.id for doc, _ in results], ["1:1"])
        self.assertEqual(
            deployment.db.mmr_calls,
            [([0.25, 0.75], 2, 100, 0.85)],
        )

    def test_candidate_depth_must_cover_requested_results(self):
        deployment = deployment_with([(FakeDocument(1, 1), 1.0)])
        with self.assertRaisesRegex(ValueError, "fetch_k must be at least k"):
            deployment.search("best_match", "query", k=50, fetch_k=20)
        self.assertEqual(deployment.embeddings.calls, [])
        self.assertEqual(deployment.db.similarity_calls, [])

    def test_equal_scores_are_deterministic(self):
        candidates = [
            (FakeDocument(1, 2), 0.5),
            (FakeDocument(1, 1), 0.5),
            (FakeDocument(2, 1), 0.5),
        ]
        deployment = deployment_with(candidates)
        first = deployment.search("best_match", "query", k=3, fetch_k=20)
        second = deployment.search("best_match", "query", k=3, fetch_k=20)
        self.assertEqual([doc.id for doc, _ in first], ["1:2", "1:1", "2:1"])
        self.assertEqual(first, second)

    def test_similarity_parent_configuration_rolls_back_exactly(self):
        candidates = [
            (FakeDocument(1, index), 1.0 - index / 100) for index in range(1, 6)
        ]
        deployment = deployment_with(candidates)
        parent = deployment.search("best_match", "query", k=5)
        rollback = deployment.search(
            "best_match",
            "query",
            k=5,
            retrieval_mode="similarity",
            fetch_k=50,
            mmr_lambda=1.0,
        )
        self.assertEqual(parent, rollback)
        self.assertEqual(deployment.db.similarity_calls, [([0.25, 0.75], 50)] * 2)

    def test_invalid_method_and_parameters_are_rejected(self):
        deployment = deployment_with([(FakeDocument(1, 1), 1.0)])
        with self.assertRaisesRegex(ValueError, "Unknown search method"):
            deployment.search("unknown", "query")
        with self.assertRaisesRegex(ValueError, "positive integer"):
            deployment.search("best_match", "query", k=True)
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            deployment.search("best_match", "query", k=1, mmr_lambda=math.inf)


if __name__ == "__main__":
    unittest.main()
