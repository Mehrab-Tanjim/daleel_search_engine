from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT / "evaluation")):
    if path not in sys.path:
        sys.path.insert(0, path)

import build_vector_database
import m05_spans
from test_ai_representation import FakeDocument, deployment_with
from test_sentence_spans import FakeTokenizer


class M05ConfigurationTests(unittest.TestCase):
    def test_configs_are_canonical_and_hashes_are_stable(self):
        for name, config in m05_spans.CONFIGS.items():
            with self.subTest(name=name):
                canonical = m05_spans.canonical_config(config)
                self.assertEqual(canonical["semantic_ai"], False)
                self.assertEqual(canonical["semantic_en_tafsir"], False)
                self.assertEqual(canonical["extra_translation_id"], "none")
                self.assertEqual(canonical["representation_scope"], "translation")
                self.assertEqual(
                    m05_spans.config_sha256(config),
                    m05_spans.config_sha256(dict(reversed(list(config.items())))),
                )

    def test_promotion_config_rejects_other_representations(self):
        invalid = (
            {"semantic_ai": True},
            {"semantic_en_tafsir": True},
            {"extra_translation_id": "pickthall_1930"},
            {"representation_scope": "ai"},
            {"span_mode": "other"},
            {"span_sentences": 3},
        )
        for override in invalid:
            with self.subTest(override=override), self.assertRaises(ValueError):
                m05_spans.canonical_config({**m05_spans.CONFIGS["parent_translation"], **override})


class M05ExportTests(unittest.TestCase):
    def test_export_retains_whole_documents_and_audits_both_span_sizes(self):
        corpus = {
            "1:1": "Rule: do not proceed without permission. Stop only when safe.",
            "2:255": " ".join(f"word{index}" for index in range(40)) + ".",
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            manifest = m05_spans.export_representation(corpus, output, FakeTokenizer(), max_tokens=8)
            manifest_path = output / "representation_manifest.json"
            self.assertEqual(json.loads(manifest_path.read_text(encoding="utf8")), manifest)
            whole = m05_spans.frozen.read_jsonl(output / "runtime" / "translation_whole_docs.jsonl")
            self.assertEqual([row["ayah"] for row in whole], ["1:1", "2:255"])
            self.assertEqual([row["text"] for row in whole], [corpus["1:1"], corpus["2:255"]])
            for count in (1, 2):
                rows = m05_spans.frozen.read_jsonl(output / "runtime" / f"translation_sentence_spans_{count}.jsonl")
                audit = manifest["outputs"][f"sentences_{count}"]["audit"]
                self.assertTrue(audit["passed"])
                self.assertEqual(audit["parents"], 2)
                self.assertEqual(audit["whole_documents"], 2)
                self.assertLessEqual(audit["max_span_tokens"], 8)
                self.assertEqual(
                    [row["id"] for row in rows if row["span_kind"] == "whole"],
                    ["translation:1:1:whole", "translation:2:255:whole"],
                )
                self.assertTrue(all(row["text"] for row in rows))
            with mock.patch.object(m05_spans, "FROZEN_CORPUS_VERSES", len(corpus)), mock.patch.object(
                m05_spans.frozen, "EXPECTED_CORPUS_SHA256", m05_spans.frozen.corpus_hash(corpus)
            ):
                self.assertEqual(m05_spans.load_representation_manifest(manifest_path), manifest)

    def test_manifest_rejects_changed_or_incomplete_outputs(self):
        corpus = {"1:1": "First sentence. Second sentence."}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            manifest = m05_spans.export_representation(corpus, output, FakeTokenizer(), max_tokens=8)
            manifest_path = output / "representation_manifest.json"
            exported = output / "runtime" / "translation_sentence_spans_1.jsonl"
            original = exported.read_bytes()
            exported.write_bytes(original + b"\n")
            with mock.patch.object(m05_spans, "FROZEN_CORPUS_VERSES", len(corpus)), mock.patch.object(
                m05_spans.frozen, "EXPECTED_CORPUS_SHA256", m05_spans.frozen.corpus_hash(corpus)
            ), self.assertRaisesRegex(ValueError, "missing or changed"):
                m05_spans.load_representation_manifest(manifest_path)
            exported.write_bytes(original)
            del manifest["outputs"]["sentences_2"]
            m05_spans.frozen.write_json(manifest_path, manifest)
            with mock.patch.object(m05_spans, "FROZEN_CORPUS_VERSES", len(corpus)), mock.patch.object(
                m05_spans.frozen, "EXPECTED_CORPUS_SHA256", m05_spans.frozen.corpus_hash(corpus)
            ), self.assertRaisesRegex(ValueError, "incomplete"):
                m05_spans.load_representation_manifest(manifest_path)

    def test_empty_corpus_text_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, "empty source"):
            m05_spans.export_representation({"1:1": "   "}, Path(directory), FakeTokenizer())


class M05EvidenceTests(unittest.TestCase):
    def test_legacy_whole_document_offsets_exclude_passage_prefix(self):
        doc = FakeDocument(1, 1, text="passage: Complete verse.")
        self.assertEqual(m05_spans._result_offsets(doc), [0, len("Complete verse.")])
        doc.page_content = "search_document: Complete verse."
        self.assertEqual(m05_spans._result_offsets(doc), [0, len("Complete verse.")])
        doc.metadata.update({"span_start": 2, "span_end": 10})
        self.assertEqual(m05_spans._result_offsets(doc), [2, 10])

    def test_execute_config_records_source_relative_offsets(self):
        doc = FakeDocument(1, 1, text="passage: Complete verse.")
        deployment = deployment_with([(doc, 0.9)])
        execution = m05_spans.execute_config(
            deployment,
            [{"id": "q1", "query": "question"}],
            m05_spans.CONFIGS["parent_translation"],
        )
        self.assertEqual(execution["runs"]["q1"]["span_offsets"], [[0, len("Complete verse.")]])

    def test_loaded_run_names_map_to_explicit_frozen_configs(self):
        selection = {"selected_name": "translation_sentences_2"}
        self.assertEqual(m05_spans.loaded_config_name("candidate", selection), "translation_sentences_2")
        self.assertEqual(m05_spans.loaded_config_name("off", selection), "parent_translation")
        self.assertEqual(m05_spans.loaded_config_name("sentences_1", selection), "translation_sentences_1")
        artifacts = {"translation_sentences_2": {"index.faiss": {"sha256": "abc"}}}
        selection["index_artifacts"] = artifacts
        self.assertTrue(m05_spans.active_index_artifact_valid(
            {"index_artifacts": artifacts}, selection, "candidate"
        ))
        self.assertFalse(m05_spans.active_index_artifact_valid(
            {"index_artifacts": {"translation_sentences_1": artifacts["translation_sentences_2"]}},
            selection,
            "candidate",
        ))

    def test_source_attribution_requires_matching_stable_span_ids(self):
        report = {
            "config": m05_spans.canonical_config(m05_spans.CONFIGS["translation_sentences_1"]),
            "runs": {
                "q1": {
                    "refs": ["1:1"],
                    "source_fields": [["translation"]],
                    "span_kinds": ["sentence"],
                    "span_offsets": [[0, 8]],
                    "span_sentences": [1],
                    "chunk_ids": ["translation:1:1:s0001"],
                }
            },
        }
        self.assertTrue(m05_spans.source_attribution_valid(report))
        for mutation in (
            {"chunk_ids": ["translation:1:2:s0001"]},
            {"chunk_ids": ["translation:1:1:whole"]},
            {"span_offsets": [[0, True]]},
            {"span_sentences": [2]},
        ):
            with self.subTest(mutation=mutation):
                changed = copy.deepcopy(report)
                changed["runs"]["q1"].update(mutation)
                self.assertFalse(m05_spans.source_attribution_valid(changed))


class M05FrozenEvidenceTests(unittest.TestCase):
    def test_determinism_compares_span_ids_offsets_and_sizes(self):
        run = {
            "refs": ["1:1"],
            "source_fields": [["translation"]],
            "span_kinds": ["sentence"],
            "chunk_ids": ["translation:1:1:s0001"],
            "span_offsets": [[0, 8]],
            "span_sentences": [2],
        }
        report = {"runs": {"q1": run}}
        repeat = {"runs": {"q1": copy.deepcopy(run)}}
        self.assertTrue(m05_spans._determinism_matches(report, repeat, "q1"))
        for field, value in (
            ("chunk_ids", ["translation:1:1:s0002"]),
            ("span_offsets", [[1, 8]]),
            ("span_sentences", [1]),
        ):
            with self.subTest(field=field):
                changed = copy.deepcopy(repeat)
                changed["runs"]["q1"][field] = value
                self.assertFalse(m05_spans._determinism_matches(report, changed, "q1"))
                self.assertNotEqual(m05_spans._execution_digest(report), m05_spans._execution_digest(changed))

    def test_frozen_code_hash_set_includes_transitive_dependencies(self):
        hashes = m05_spans.m05_code_hashes()
        self.assertEqual(set(hashes), set(m05_spans.FROZEN_CODE_FILES))
        for dependency in (
            "evaluation/islamqa_frozen.py",
            "evaluation/m03_tafsir.py",
            "evaluation/m04_translation.py",
            "evaluation/representations.py",
        ):
            self.assertIn(dependency, hashes)
        m05_spans.verify_frozen_code({"code_sha256": hashes})
        for changed in (
            {key: value for key, value in hashes.items() if key != "evaluation/m03_tafsir.py"},
            {**hashes, "unexpected.py": "0" * 64},
        ):
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, "hash set"):
                m05_spans.verify_frozen_code({"code_sha256": changed})

    def test_frozen_scorer_hash_rejects_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            scorer = package / "evaluate_search.py"
            scorer.write_text("def score_group():\n    return {}\n", encoding="utf8")
            selection = {"scorer_sha256": m05_spans.frozen.text_sha256(scorer)}
            m05_spans.verify_frozen_scorer(selection, package)
            scorer.write_text("def score_group():\n    return {'changed': True}\n", encoding="utf8")
            with self.assertRaisesRegex(ValueError, "scorer changed"):
                m05_spans.verify_frozen_scorer(selection, package)

    def test_validation_evidence_rehashes_artifacts_and_recomputes_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            cases = [{"id": "q1"}]
            corpus = {"1:1": "source"}
            fixture = {
                "fixture_sha256": m05_spans.frozen.EXPECTED_VALIDATION_FIXTURE_SHA256,
                "corpus": {"sha256": m05_spans.frozen.corpus_hash(corpus)},
            }
            reports = {}
            summaries = {}
            artifacts = {}
            for name, config in m05_spans.CONFIGS.items():
                run = {
                    "refs": ["1:1"],
                    "source_fields": [["translation"]],
                    "span_kinds": ["whole" if config["span_mode"] == "whole" else "sentence"],
                    "chunk_ids": [f"translation:1:1:{'whole' if config['span_mode'] == 'whole' else 's0001'}"],
                    "span_offsets": [[0, 6]],
                    "span_sentences": [config["span_sentences"]],
                }
                reports[name] = {
                    "config": config,
                    "parent_commit": m05_spans.PARENT_COMMIT,
                    "representation_manifest_sha256": "manifest",
                    "runs": {"q1": run},
                }
                summaries[name] = {
                    "name": name,
                    "config": config,
                    "config_sha256": m05_spans.config_sha256(config),
                }
                artifact = directory / f"{name}.json"
                artifact.write_text(json.dumps(reports[name]), encoding="utf8")
                artifacts[name] = {
                    "file": artifact.name,
                    "sha256": m05_spans.frozen.sha256_file(artifact),
                }
            selected = summaries["translation_sentences_2"]
            parent = summaries["parent_translation"]
            gates = {"translation_sentences_1": False, "translation_sentences_2": True}
            digest = m05_spans._execution_digest({"runs": reports[selected["name"]]["runs"]})
            selection = {
                "experiment": "M05",
                "split": "validation",
                "test_accessed": False,
                "representation_manifest_sha256": "manifest",
                "candidates": summaries,
                "candidate_validation_gates": gates,
                "selected_name": selected["name"],
                "selected_config": selected["config"],
                "selected_config_sha256": selected["config_sha256"],
                "selected_changed": selected["config_sha256"] != parent["config_sha256"],
                "validation_gate_passed": True,
                "determinism_check": {
                    "queries": 1,
                    "exact_queries": 1,
                    "passed": True,
                    "selected_runs_sha256": digest,
                    "repeat_runs_sha256": digest,
                },
                "validation_ranking_artifacts": artifacts,
            }
            def name_for(config):
                return next(name for name, candidate in m05_spans.CONFIGS.items() if candidate == config)

            def load_report(path, *args):
                return copy.deepcopy(reports[Path(path).stem]), {"q1": ["1:1"]}

            def summarize(report, *args):
                return copy.deepcopy(summaries[name_for(report["config"])])

            with mock.patch.object(m05_spans, "FROZEN_CORPUS_VERSES", 1), mock.patch.object(
                m05_spans, "verify_frozen_code"
            ), mock.patch.object(m05_spans, "verify_frozen_scorer"), mock.patch.object(
                m05_spans.frozen, "load_fixture", return_value=(cases, fixture)
            ), mock.patch.object(m05_spans.frozen, "load_corpus", return_value=corpus), mock.patch.object(
                m05_spans.frozen, "scorer_module", return_value=object()
            ), mock.patch.object(m05_spans.frozen, "score_run", return_value={}), mock.patch.object(
                m05_spans.frozen, "load_run", side_effect=load_report
            ), mock.patch.object(m05_spans, "source_attribution_valid", return_value=True), mock.patch.object(
                m05_spans, "_headline_summary", side_effect=summarize
            ), mock.patch.object(m05_spans, "select_candidate", return_value=selected), mock.patch.object(
                m05_spans.frozen, "validation_gate", side_effect=lambda parent, candidate: gates[candidate["name"]]
            ):
                m05_spans.verify_validation_evidence(selection, directory / "selection.json", directory / "corpus.db", directory, 0, 4213)
                artifact = directory / artifacts["parent_translation"]["file"]
                artifact.write_bytes(artifact.read_bytes() + b"\n")
                with self.assertRaisesRegex(ValueError, "artifact changed"):
                    m05_spans.verify_validation_evidence(selection, directory / "selection.json", directory / "corpus.db", directory, 0, 4213)
                artifact.write_text(json.dumps(reports["parent_translation"]), encoding="utf8")
                changed = copy.deepcopy(selection)
                changed["selected_name"] = "translation_sentences_1"
                with self.assertRaisesRegex(ValueError, "selection does not match"):
                    m05_spans.verify_validation_evidence(changed, directory / "selection.json", directory / "corpus.db", directory, 0, 4213)


class M05LeaderboardTests(unittest.TestCase):
    def test_m05_column_is_appended_after_m04(self):
        summary = {
            "runs": {
                **{
                    name: {"headline": {slice_name: {"metrics": {}, "counts": {}} for slice_name in ("all", "useful", "direct")}}
                    for name, _ in m05_spans.frozen.RUN_COLUMNS
                },
                **{
                    name: {
                        "leaderboard_label": label,
                        "headline": {slice_name: {"metrics": {}, "counts": {}} for slice_name in ("all", "useful", "direct")},
                    }
                    for name, label in (
                        ("m01", "M01 similarity"),
                        ("m02", "M02 translation"),
                        ("m03", "M03 translation"),
                        ("m04", "M04 translation"),
                        ("m05", "M05 sentences-1"),
                    )
                },
            }
        }
        header = m05_spans.frozen.markdown_leaderboard(summary).splitlines()[0]
        self.assertIn("M04 translation | M05 sentences-1", header)

    def test_frozen_cli_registers_all_m05_commands(self):
        environment = os.environ.copy()
        environment["EVAL_PROFILE"] = "pr2_frozen_v1"
        result = subprocess.run(
            [sys.executable, str(ROOT / "run_benchmark.py"), "--help"],
            cwd=ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf8",
        )
        for command in ("m05-export", "m05-sweep", "m05-config", "m05-report"):
            self.assertIn(command, result.stdout)


class ProductionBuilderTests(unittest.TestCase):
    def _write_datasets(self, directory: Path):
        from langchain_core.documents import Document

        datasets = directory / "datasets"
        datasets.mkdir()
        quran = [Document(
            page_content="Rule: do not omit (this). Stop when safe.",
            metadata={"SurahNo": "1", "AyahNo": "1"},
        )]
        hadith = [Document(page_content="Hadith text.", metadata={"source": "test"})]
        for doctype in ("original", "preprocessed"):
            with (datasets / f"{doctype}_quran_docs.pkl").open("wb") as handle:
                pickle.dump(quran, handle)
            with (datasets / f"{doctype}_hadith_docs.pkl").open("wb") as handle:
                pickle.dump(hadith, handle)

    def test_sentence_builder_uses_separate_index_and_applies_nomic_prefix_once(self):
        embeddings = mock.Mock()
        embeddings.client.tokenizer = FakeTokenizer()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            self._write_datasets(directory)
            previous = os.getcwd()
            try:
                os.chdir(directory)
                with mock.patch.dict(os.environ, {"SPAN_MODE": "sentences", "SPAN_SENTENCES": "2"}, clear=False), mock.patch.object(
                    build_vector_database, "HuggingFaceEmbeddings", return_value=embeddings
                ), mock.patch.object(build_vector_database.os, "makedirs"), mock.patch.object(
                    build_vector_database, "save_vectorstore"
                ) as save:
                    build_vector_database.build_vd("nomic-ai/nomic-embed-text-v2-moe", "cpu")
            finally:
                os.chdir(previous)
        self.assertEqual(save.call_count, 4)
        quran_calls = [call for call in save.call_args_list if call.args[2].endswith("quran-sentences-2")]
        hadith_calls = [call for call in save.call_args_list if call.args[2].endswith("/hadith")]
        self.assertEqual(len(quran_calls), 2)
        self.assertEqual(len(hadith_calls), 2)
        for call in quran_calls:
            documents = call.args[0]
            self.assertEqual(documents[0].metadata["span_kind"], "whole")
            self.assertEqual(documents[0].metadata["surah"], 1)
            self.assertEqual(documents[0].metadata["ayah"], 1)
            self.assertTrue(all(doc.page_content.startswith("search_document: ") for doc in documents))
            self.assertTrue(all(not doc.page_content.startswith("search_document: search_document: ") for doc in documents))
            self.assertTrue(any(doc.metadata["span_kind"] == "sentence" for doc in documents))
            self.assertTrue(all("original_text" not in doc.metadata for doc in documents))
        for call in hadith_calls:
            self.assertEqual(call.args[0][0].page_content, "search_document: Hadith text.")

    def test_sentence_builder_requires_embedding_tokenizer(self):
        embeddings = mock.Mock()
        embeddings.client.tokenizer = None
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            self._write_datasets(directory)
            previous = os.getcwd()
            try:
                os.chdir(directory)
                with mock.patch.dict(os.environ, {"SPAN_MODE": "sentences", "SPAN_SENTENCES": "1"}, clear=False), mock.patch.object(
                    build_vector_database, "HuggingFaceEmbeddings", return_value=embeddings
                ), mock.patch.object(build_vector_database.os, "makedirs"), self.assertRaisesRegex(
                    ValueError, "requires the embedding tokenizer"
                ):
                    build_vector_database.build_vd("sentence-transformers/all-mpnet-base-v2", "cpu")
            finally:
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
