from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any
from unittest import mock

try:
    from . import islamqa_frozen as frozen
    from . import m03_tafsir
    from . import m04_translation
except ImportError:
    import islamqa_frozen as frozen
    import m03_tafsir
    import m04_translation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import sentence_spans
import utils


HERE = Path(__file__).resolve().parent
DEFAULT_M05_VALIDATION = HERE / "results" / "m05_spans" / "validation"
DEFAULT_M05_TEST = HERE / "results" / "m05_spans" / "test"
DEFAULT_M04_TEST = HERE / "results" / "m04_translation" / "test"
PARENT_COMMIT = "525eb0a3dfafd9c6359d30a7e9c5e970efc80aaf"
FROZEN_CORPUS_VERSES = 6236
LONG_DOCUMENT_TOKENS = 128
FROZEN_CODE_FILES = (
    "sentence_spans.py", "utils.py", "prepare_documents.py", "build_vector_database.py",
    "app.py", "dev_app.py", "run_benchmark.py", "evaluation/islamqa_frozen.py",
    "evaluation/m03_tafsir.py", "evaluation/m04_translation.py", "evaluation/representations.py",
    "evaluation/m05_spans.py", "evaluation/test_sentence_spans.py", "evaluation/test_m05_spans.py",
)
DETERMINISM_FIELDS = (
    "refs", "source_fields", "span_kinds", "chunk_ids", "span_offsets", "span_sentences",
)
CONFIGS = {
    "parent_translation": {
        "semantic_ai": False,
        "semantic_en_tafsir": False,
        "extra_translation_id": "none",
        "representation_scope": "translation",
        "span_mode": "whole",
        "span_sentences": 1,
    },
    "translation_sentences_1": {
        "semantic_ai": False,
        "semantic_en_tafsir": False,
        "extra_translation_id": "none",
        "representation_scope": "translation",
        "span_mode": "sentences",
        "span_sentences": 1,
    },
    "translation_sentences_2": {
        "semantic_ai": False,
        "semantic_en_tafsir": False,
        "extra_translation_id": "none",
        "representation_scope": "translation",
        "span_mode": "sentences",
        "span_sentences": 2,
    },
}


def canonical_config(config: dict[str, Any]) -> dict[str, Any]:
    base = m04_translation.canonical_config(config)
    mode, count = sentence_spans.canonical_span_config(config["span_mode"], config["span_sentences"])
    if base["semantic_ai"] or base["semantic_en_tafsir"] or base["extra_translation_id"] != "none":
        raise ValueError("M05 promotion configs require AI, tafsir, and extra translation to be off")
    if base["representation_scope"] != "translation":
        raise ValueError("M05 promotion configs require the translation representation")
    return {**base, "span_mode": mode, "span_sentences": count}


def config_sha256(config: dict[str, Any]) -> str:
    payload = json.dumps(canonical_config(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf8")).hexdigest()


def _token_summary(rows: list[dict[str, Any]], whole_rows: list[dict[str, Any]], max_tokens: int = 512) -> dict[str, Any]:
    spans = [row for row in rows if row["span_kind"] == "sentence"]
    counts = sorted(row["token_count"] for row in spans)
    whole_counts = sorted(row["token_count"] for row in whole_rows)
    long_refs = [row["ayah"] for row in whole_rows if row["token_count"] > LONG_DOCUMENT_TOKENS]
    return {
        "counter": "AutoTokenizer.encode with passage prefix and special tokens",
        "max_span_tokens": max_tokens,
        "span_documents": len(spans),
        "span_min_tokens": min(counts),
        "span_median_tokens": statistics.median(counts),
        "span_p95_tokens": counts[max(0, math.ceil(0.95 * len(counts)) - 1)],
        "span_max_tokens": max(counts),
        "spans_over_budget": sum(count > max_tokens for count in counts),
        "whole_documents": len(whole_rows),
        "whole_over_budget": sum(count > 512 for count in whole_counts),
        "whole_max_tokens": max(whole_counts),
        "long_document_threshold_tokens": LONG_DOCUMENT_TOKENS,
        "long_document_ayahs": long_refs,
        "long_documents": len(long_refs),
    }


def export_representation(
    corpus: dict[str, str],
    output_dir: Path,
    tokenizer: Any,
    max_tokens: int = 512,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    refs = sorted(corpus, key=lambda ref: tuple(map(int, ref.split(":"))))
    whole_rows = []
    for ref in refs:
        text = corpus[ref]
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Cannot export empty source text: {ref}")
        surah, ayah = (int(part) for part in ref.split(":"))
        whole_rows.append({
            "id": f"translation:{ref}:whole",
            "ayah": ref,
            "text": text,
            "field": "translation",
            "source": "saheeh_international",
            "metadata": {"surah": surah, "ayah": ayah},
            "span_kind": "whole",
            "span_start": 0,
            "span_end": len(text),
            "chunk": 0,
            "sentence_start": None,
            "sentence_end": None,
            "sentence_count": len(sentence_spans.sentence_ranges(text)),
            "span_sentences": 1,
            "source_text_sha256": sentence_spans.source_text_sha256(text),
            "token_count": sentence_spans.token_count(tokenizer, text),
        })
    whole_path = output_dir / "runtime" / "translation_whole_docs.jsonl"
    frozen.write_jsonl(whole_path, whole_rows)
    whole_ids = [row["id"] for row in whole_rows]
    if len(whole_ids) != len(set(whole_ids)) or set(row["ayah"] for row in whole_rows) != set(corpus):
        raise ValueError("Whole-document export has duplicate or missing parents")
    for row in whole_rows:
        source = corpus[row["ayah"]]
        if row["text"] != source or row["span_start"] != 0 or row["span_end"] != len(source):
            raise ValueError(f"Whole-document export is not source-grounded: {row['ayah']}")
        if row["source_text_sha256"] != sentence_spans.source_text_sha256(source):
            raise ValueError(f"Whole-document export source hash mismatch: {row['ayah']}")
    outputs: dict[str, Any] = {
        "whole": {
            "file": whole_path.relative_to(output_dir).as_posix(),
            "sha256": frozen.sha256_file(whole_path),
            "bytes": whole_path.stat().st_size,
            "rows": len(whole_rows),
            "audit": {
                "passed": True,
                "parents": len(whole_rows),
                "exact_source_substrings": True,
                "stable_unique_ids": True,
                "no_empty_spans": True,
            },
        }
    }
    for count in (1, 2):
        rows = []
        for ref in refs:
            surah, ayah = (int(part) for part in ref.split(":"))
            rows.extend(sentence_spans.build_span_rows(
                corpus[ref],
                {"surah": surah, "ayah": ayah},
                "translation",
                "saheeh_international",
                count,
                tokenizer,
                max_tokens=max_tokens,
            ))
        audit = sentence_spans.audit_span_rows(rows, corpus, max_tokens=max_tokens)
        path = output_dir / "runtime" / f"translation_sentence_spans_{count}.jsonl"
        frozen.write_jsonl(path, rows)
        outputs[f"sentences_{count}"] = {
            "file": path.relative_to(output_dir).as_posix(),
            "sha256": frozen.sha256_file(path),
            "bytes": path.stat().st_size,
            "rows": len(rows),
            "audit": audit,
            "tokens": _token_summary(rows, whole_rows, max_tokens),
        }
    manifest = {
        "experiment": "M05",
        "representation": "sentence_spans",
        "display_field": "translation",
        "segmenter": sentence_spans.SEGMENTER_VERSION,
        "segmenter_sha256": frozen.sha256_file(ROOT / "sentence_spans.py"),
        "corpus": {"verses": len(corpus), "sha256": frozen.corpus_hash(corpus)},
        "source": {
            "field": "translation",
            "source": "saheeh_international",
            "policy": "Segment the original accepted corpus text; do not use destructive preprocess_text output.",
        },
        "counts": {
            "parents": len(corpus),
            "whole_documents": len(whole_rows),
            "sentences_1_rows": outputs["sentences_1"]["rows"],
            "sentences_2_rows": outputs["sentences_2"]["rows"],
        },
        "outputs": outputs,
        "selection_sources": {
            "dataset_commit": frozen.DATASET_COMMIT,
            "test_sources_used": False,
            "policy": "Representation export is corpus-only; configuration selection uses validation only.",
        },
    }
    frozen.write_json(output_dir / "representation_manifest.json", manifest)
    return manifest


def export_spans(args) -> None:
    from transformers import AutoTokenizer

    corpus = frozen.load_corpus(args.corpus_db)
    if len(corpus) != FROZEN_CORPUS_VERSES or frozen.corpus_hash(corpus) != frozen.EXPECTED_CORPUS_SHA256:
        raise ValueError("Corpus does not match the frozen fixture")
    tokenizer = AutoTokenizer.from_pretrained(frozen.model_snapshot(), local_files_only=True)
    manifest = export_representation(corpus, args.output_dir, tokenizer, max_tokens=512)
    frozen.write_json(args.output_dir / "representation_manifest.json", manifest)
    print(json.dumps({
        "manifest": str(args.output_dir / "representation_manifest.json"),
        "counts": manifest["counts"],
        "audits": {name: value["audit"] for name, value in manifest["outputs"].items() if name != "whole"},
        "long_documents": manifest["outputs"]["sentences_1"]["tokens"]["long_documents"],
    }, indent=2))


def load_representation_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf8"))
    if manifest.get("experiment") != "M05" or manifest.get("representation") != "sentence_spans":
        raise ValueError("Invalid M05 representation manifest")
    if manifest.get("display_field") != "translation" or manifest.get("segmenter") != sentence_spans.SEGMENTER_VERSION:
        raise ValueError("M05 display or segmenter contract changed")
    if manifest["corpus"] != {"verses": FROZEN_CORPUS_VERSES, "sha256": frozen.EXPECTED_CORPUS_SHA256}:
        raise ValueError("M05 representation corpus hash changed")
    if manifest.get("segmenter_sha256") != frozen.sha256_file(ROOT / "sentence_spans.py"):
        raise ValueError("M05 segmenter implementation changed")
    if manifest.get("selection_sources", {}).get("test_sources_used") is not False:
        raise ValueError("M05 export must not use test sources")
    if set(manifest["outputs"]) != {"whole", "sentences_1", "sentences_2"}:
        raise ValueError("M05 representation outputs are incomplete")
    for name, output in manifest["outputs"].items():
        output_path = path.parent / output["file"]
        if not output_path.exists() or frozen.sha256_file(output_path) != output["sha256"]:
            raise ValueError(f"M05 representation output missing or changed: {name}")
        rows = frozen.read_jsonl(output_path)
        if output_path.stat().st_size != output["bytes"] or len(rows) != output["rows"]:
            raise ValueError(f"M05 representation output size or row count mismatch: {name}")
        if output["audit"].get("passed") is not True:
            raise ValueError(f"M05 span audit failed: {name}")
        if name == "whole":
            refs = [row.get("ayah") for row in rows]
            if (
                len(refs) != FROZEN_CORPUS_VERSES or len(set(refs)) != len(refs)
                or any(row.get("span_kind") != "whole" or row.get("span_start") != 0 or row.get("span_end") != len(row.get("text", "")) for row in rows)
            ):
                raise ValueError("M05 whole-document output audit failed")
        else:
            source_texts = {row["ayah"]: row["text"] for row in rows if row.get("span_kind") == "whole"}
            if sentence_spans.audit_span_rows(rows, source_texts) != output["audit"]:
                raise ValueError(f"M05 span audit does not match exported rows: {name}")
    return manifest


def _document_metadata(row: dict[str, Any]) -> dict[str, Any]:
    surah, ayah = (int(part) for part in row["ayah"].split(":"))
    keys = (
        "span_kind", "span_start", "span_end", "chunk", "sentence_start", "sentence_end",
        "sentence_count", "span_sentences", "source_text_sha256", "token_count",
    )
    return {
        "surah": surah,
        "ayah": ayah,
        "field": row["field"],
        "source": row["source"],
        **{key: row[key] for key in keys},
    }


def _load_span_index(deployment, path: Path, output: Path, manifest: dict[str, Any], count: int) -> Any:
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document

    output_info = manifest["outputs"][f"sentences_{count}"]
    if frozen.sha256_file(output) != output_info["sha256"]:
        raise ValueError(f"M05 sentence-{count} export hash mismatch")
    rows = frozen.read_jsonl(output)
    if len(rows) != output_info["rows"]:
        raise ValueError(f"M05 sentence-{count} export count mismatch")
    expected = {row["id"]: row for row in rows}
    if len(expected) != len(rows):
        raise ValueError(f"M05 sentence-{count} export has duplicate IDs")
    source_texts = {row["ayah"]: row["text"] for row in rows if row.get("span_kind") == "whole"}
    if sentence_spans.audit_span_rows(rows, source_texts) != output_info["audit"]:
        raise ValueError(f"M05 sentence-{count} export audit mismatch")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not (path / "index.faiss").exists():
        documents = [
            Document(page_content="passage: " + row["text"], metadata=_document_metadata(row), id=row["id"])
            for row in rows
        ]
        database = FAISS.from_documents(
            documents,
            deployment.embeddings,
            normalize_L2=True,
            distance_strategy="MAX_INNER_PRODUCT",
        )
        database.save_local(str(path))
    database = FAISS.load_local(
        str(path), deployment.embeddings, normalize_L2=True, allow_dangerous_deserialization=True
    )
    if database.index.ntotal != len(rows) or set(database.index_to_docstore_id.values()) != set(expected):
        raise ValueError(f"M05 sentence-{count} index size or ID mismatch")
    for doc_id, row in expected.items():
        document = database.docstore.search(doc_id)
        if document is None or document.page_content != "passage: " + row["text"] or document.metadata != _document_metadata(row):
            raise ValueError(f"M05 sentence-{count} index content mismatch: {row['ayah']}/{row['id']}")
    return database


def index_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def index_artifacts(path: Path) -> dict[str, dict[str, Any]]:
    return {
        item.relative_to(path).as_posix(): {"sha256": frozen.sha256_file(item), "bytes": item.stat().st_size}
        for item in sorted(path.rglob("*")) if item.is_file()
    }


def load_or_build_indexes(args, cases, fixture_manifest, manifest, config_names):
    with mock.patch.dict(os.environ, {
        "SEMANTIC_AI": "false", "SEMANTIC_EN_TAFSIR": "false", "EXTRA_TRANSLATION_ID": "none",
        "SPAN_MODE": "whole", "SPAN_SENTENCES": "1",
    }, clear=False):
        deployment, parent_path, snapshot = frozen.load_or_build_dense_index(args, cases, fixture_manifest)
    paths = {"parent_translation": parent_path}
    databases = {"parent_translation": deployment.db}
    for count in sorted({CONFIGS[name]["span_sentences"] for name in config_names if CONFIGS[name]["span_mode"] == "sentences"}):
        name = f"translation_sentences_{count}"
        path = args.runtime_dir / "e5-small" / f"quran-sentences-{count}"
        output = args.spans_output_1 if count == 1 else args.spans_output_2
        databases[name] = _load_span_index(deployment, path, output, manifest, count)
        paths[name] = path
    return deployment, paths, snapshot, databases


def activate_config(deployment, databases, name):
    config = canonical_config(CONFIGS[name])
    database = databases[name]
    deployment.db = database
    deployment.db_docstore_id_to_index = dict(zip(database.index_to_docstore_id.values(), database.index_to_docstore_id.keys()))
    deployment.span_mode = config["span_mode"]
    deployment.span_sentences = config["span_sentences"]
    if hasattr(deployment, "translation_docs_by_ayah"):
        del deployment.translation_docs_by_ayah


def _result_offsets(doc):
    metadata = doc.metadata
    if "span_start" in metadata or "span_end" in metadata:
        start, end = metadata.get("span_start"), metadata.get("span_end")
    else:
        content = doc.page_content
        for prefix in ("passage: ", "search_document: "):
            if content.startswith(prefix):
                content = content[len(prefix):]
                break
        start, end = 0, len(content)
    if (
        isinstance(start, bool) or isinstance(end, bool)
        or not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end
    ):
        raise ValueError("Retrieved document has invalid source-relative span offsets")
    return [start, end]


def execute_config(deployment, cases, config):
    config = canonical_config(config)
    runs = {}
    for index, case in enumerate(cases):
        started = time.perf_counter()
        results = deployment.search(
            "best_match", "query: " + case["query"], k=50,
            retrieval_mode="similarity", fetch_k=50, mmr_lambda=1.0,
            semantic_ai=False, semantic_en_tafsir=False, extra_translation_id="none",
            representation_scope="translation", span_mode=config["span_mode"],
            span_sentences=config["span_sentences"],
        )
        elapsed = round((time.perf_counter() - started) * 1000, 3)
        refs = [f"{doc.metadata['surah']}:{doc.metadata['ayah']}" for doc, _ in results]
        if len(refs) != len(set(refs)) or len(refs) > 50:
            raise ValueError(f"M05 duplicate or over-depth result: {case['id']}")
        fields = [doc.metadata.get("field", "translation") for doc, _ in results]
        if set(fields) != {"translation"}:
            raise ValueError(f"M05 source-field leakage: {case['id']}")
        runs[case["id"]] = {
            "refs": refs,
            "source_fields": [[field] for field in fields],
            "span_kinds": [doc.metadata.get("span_kind", "whole") for doc, _ in results],
            "chunk_ids": [doc.id for doc, _ in results],
            "span_offsets": [_result_offsets(doc) for doc, _ in results],
            "span_sentences": [doc.metadata.get("span_sentences", config["span_sentences"]) for doc, _ in results],
            "ms": elapsed,
        }
        if index % 50 == 0:
            print(f"M05 {config['span_mode']}/{config['span_sentences']}: {index}/{len(cases)}", flush=True)
    return {"config": config, "result_k": 50, "runs": runs}


def config_report(args, name, cases, fixture_manifest, deployment, paths, snapshot, manifest):
    config = canonical_config(CONFIGS[name])
    execution = execute_config(deployment, cases, config)
    path = paths[name]
    return {
        "engine": "M05 offline multilingual E5-small float32 FAISS with optional sentence spans",
        "profile": "pr2_frozen_v1",
        "experiment": "M05",
        "model": frozen.E5_MODEL,
        "model_revision": frozen.E5_REVISION,
        "model_files": frozen.dense_model_files(snapshot),
        "dtype": "float32",
        "distance": "max_inner_product",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
        "field": "translation",
        "display_field": "translation",
        "source_attribution": "per-result winning representation and source span",
        "fixture_sha256": fixture_manifest["fixture_sha256"],
        "corpus": {"verses": FROZEN_CORPUS_VERSES, "sha256": fixture_manifest["corpus"]["sha256"]},
        "retrieval": {"mode": "similarity", "fetch_k": 50, "mmr_lambda": 1.0},
        "aggregation": "Fetch 50 per active representation; keep one maximum-scored result per ayah; ties prefer translation, AI, English tafsir, then extra translation, followed by numeric ayah ID.",
        "index_bytes": index_bytes(path),
        "index_artifacts": {name: index_artifacts(path)},
        "active_index": path.name,
        "peak_rss_bytes": frozen.peak_rss_bytes(),
        "representation_manifest_sha256": frozen.sha256_file(args.representation_manifest),
        "parent_commit": PARENT_COMMIT,
        **execution,
        "config_sha256": config_sha256(config),
    }


def m05_code_hashes():
    return {relative: frozen.sha256_file(ROOT / relative) for relative in FROZEN_CODE_FILES}


def verify_frozen_code(selection):
    recorded = selection.get("code_sha256")
    if not isinstance(recorded, dict) or set(recorded) != set(FROZEN_CODE_FILES):
        raise ValueError("Frozen M05 code hash set is incomplete or unexpected")
    for relative, digest in recorded.items():
        if frozen.sha256_file(ROOT / relative) != digest:
            raise ValueError(f"Frozen M05 code changed after selection: {relative}")


def _execution_digest(execution):
    payload = {
        case_id: {field: run[field] for field in DETERMINISM_FIELDS}
        for case_id, run in sorted(execution["runs"].items())
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf8")
    return hashlib.sha256(encoded).hexdigest()


def _determinism_matches(report, repeat, case_id):
    selected_run = report["runs"][case_id]
    repeated_run = repeat["runs"][case_id]
    return all(selected_run[field] == repeated_run[field] for field in DETERMINISM_FIELDS)


def verify_frozen_scorer(selection, scorer_package):
    expected = frozen.text_sha256(scorer_package / "evaluate_search.py")
    if selection.get("scorer_sha256") != expected:
        raise ValueError("Frozen M05 scorer changed after selection")


def verify_validation_evidence(selection, selection_path, corpus_db, scorer_package, bootstrap, seed):
    if selection.get("experiment") != "M05" or selection.get("split") != "validation" or selection.get("test_accessed") is not False:
        raise ValueError("M05 validation selection contract mismatch")
    verify_frozen_code(selection)
    verify_frozen_scorer(selection, scorer_package)
    validation_dir = selection_path.parent
    cases, fixture_manifest = frozen.load_fixture(validation_dir, "validation")
    if fixture_manifest["fixture_sha256"] != frozen.EXPECTED_VALIDATION_FIXTURE_SHA256:
        raise ValueError("M05 validation fixture does not match the frozen contract")
    corpus = frozen.load_corpus(corpus_db)
    if len(corpus) != FROZEN_CORPUS_VERSES or frozen.corpus_hash(corpus) != fixture_manifest["corpus"]["sha256"]:
        raise ValueError("M05 validation corpus does not match the frozen contract")
    score_group = frozen.scorer_module(scorer_package)
    artifacts = selection.get("validation_ranking_artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(CONFIGS):
        raise ValueError("M05 validation ranking artifacts are incomplete")
    reports = {}
    summaries = {}
    for name in CONFIGS:
        artifact = artifacts[name]
        path = validation_dir / artifact["file"]
        if frozen.sha256_file(path) != artifact["sha256"]:
            raise ValueError(f"M05 validation ranking artifact changed: {name}")
        report, rankings = frozen.load_run(path, cases, set(corpus), fixture_manifest)
        if (
            canonical_config(report["config"]) != canonical_config(CONFIGS[name])
            or report.get("parent_commit") != PARENT_COMMIT
            or report.get("representation_manifest_sha256") != selection.get("representation_manifest_sha256")
            or not source_attribution_valid(report)
        ):
            raise ValueError(f"M05 validation run contract mismatch: {name}")
        scores = frozen.score_run(cases, rankings, score_group)
        summary = _headline_summary(report, cases, rankings, scores, bootstrap, seed)
        summary["name"] = name
        if summary != selection["candidates"][name]:
            raise ValueError(f"M05 validation summary changed: {name}")
        reports[name] = report
        summaries[name] = summary
    parent = summaries["parent_translation"]
    candidates = [summaries["translation_sentences_1"], summaries["translation_sentences_2"]]
    selected = select_candidate(parent, candidates)
    gates = {name: frozen.validation_gate(parent, summaries[name]) for name in ("translation_sentences_1", "translation_sentences_2")}
    expected = {
        "selected_name": selected["name"],
        "selected_config": selected["config"],
        "selected_config_sha256": selected["config_sha256"],
        "selected_changed": selected["config_sha256"] != parent["config_sha256"],
        "validation_gate_passed": gates.get(selected["name"], False),
        "candidate_validation_gates": gates,
    }
    if any(selection.get(key) != value for key, value in expected.items()):
        raise ValueError("M05 validation selection does not match its evidence")
    determinism = selection.get("determinism_check", {})
    digest = _execution_digest({"runs": reports[selected["name"]]["runs"]})
    if (
        determinism.get("queries") != len(cases)
        or determinism.get("exact_queries") != len(cases)
        or determinism.get("passed") is not True
        or determinism.get("selected_runs_sha256") != digest
        or determinism.get("repeat_runs_sha256") != digest
    ):
        raise ValueError("M05 validation determinism evidence changed")
    return summaries


def _headline_summary(report, cases, rankings, scores, bootstrap, seed):
    summary = m03_tafsir._headline_summary(report, cases, rankings, scores, bootstrap, seed)
    config = report["config"]
    summary["leaderboard_label"] = "M05 whole" if config["span_mode"] == "whole" else f"M05 sentences-{config['span_sentences']}"
    return summary


def select_candidate(parent, candidates):
    return frozen.choose_validation_candidate(parent, candidates)


def run_sweep(args) -> None:
    cases, fixture_manifest = frozen.load_fixture(args.output_dir, "validation")
    if fixture_manifest["fixture_sha256"] != frozen.EXPECTED_VALIDATION_FIXTURE_SHA256:
        raise ValueError("M05 validation fixture does not match the frozen contract")
    corpus = frozen.load_corpus(args.corpus_db)
    score_group = frozen.scorer_module(args.scorer_package)
    scorer_sha256 = frozen.text_sha256(args.scorer_package / "evaluate_search.py")
    manifest = load_representation_manifest(args.representation_manifest)
    deployment, paths, snapshot, databases = load_or_build_indexes(args, cases, fixture_manifest, manifest, tuple(CONFIGS))
    summaries = {}
    scores_by_name = {}
    artifacts = {}
    for name, config in CONFIGS.items():
        activate_config(deployment, databases, name)
        report = config_report(args, name, cases, fixture_manifest, deployment, paths, snapshot, manifest)
        path = args.output_dir / "validation_runs" / f"{name}.json"
        frozen.write_json(path, report)
        rankings = frozen.validate_run(cases, report, set(corpus))
        scores = frozen.score_run(cases, rankings, score_group)
        summary = _headline_summary(report, cases, rankings, scores, args.bootstrap, args.seed)
        summary["name"] = name
        summaries[name] = summary
        scores_by_name[name] = scores
        artifacts[name] = {"file": str(path.relative_to(args.output_dir)), "sha256": frozen.sha256_file(path)}
    parent = summaries["parent_translation"]
    candidates = [summaries["translation_sentences_1"], summaries["translation_sentences_2"]]
    selected = select_candidate(parent, candidates)
    gates = {name: frozen.validation_gate(parent, summaries[name]) for name in ("translation_sentences_1", "translation_sentences_2")}
    marginal_effects = {
        name: m03_tafsir._paired_metrics(cases, scores_by_name["parent_translation"], scores, args.bootstrap, args.seed)
        for name, scores in scores_by_name.items()
    }
    activate_config(deployment, databases, selected["name"])
    repeat = execute_config(deployment, cases, selected["config"])
    selected_path = args.output_dir / "validation_runs" / f"{selected['name']}.json"
    selected_report = json.loads(selected_path.read_text(encoding="utf8"))
    exact = sum(_determinism_matches(selected_report, repeat, case["id"]) for case in cases)
    if exact != len(cases):
        raise ValueError("M05 selected configuration is not deterministic")
    determinism_digest = _execution_digest(repeat)
    selection = {
        "profile": "pr2_frozen_v1",
        "experiment": "M05",
        "split": "validation",
        "dataset": fixture_manifest,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_accessed": False,
        "selection_policy": "Only one- or two-sentence translation spans are promotion-eligible. Require higher validation useful hit@5 with no validation nDCG@10 decrease; exact ties retain the simpler one-sentence candidate, and no passing candidate retains the whole-document parent.",
        "candidates": summaries,
        "candidate_validation_gates": gates,
        "marginal_effects": marginal_effects,
        "selected_name": selected["name"],
        "selected_config": selected["config"],
        "selected_config_sha256": selected["config_sha256"],
        "selected_changed": selected["config_sha256"] != parent["config_sha256"],
        "validation_gate_passed": gates.get(selected["name"], False),
        "determinism_check": {
            "queries": len(cases),
            "exact_queries": exact,
            "passed": exact == len(cases),
            "selected_runs_sha256": determinism_digest,
            "repeat_runs_sha256": determinism_digest,
        },
        "representation_manifest": manifest,
        "representation_manifest_sha256": frozen.sha256_file(args.representation_manifest),
        "index_artifacts": {name: index_artifacts(path) for name, path in paths.items()},
        "model": {"name": frozen.E5_MODEL, "revision": frozen.E5_REVISION, "files": frozen.dense_model_files(snapshot)},
        "prefixes": {"query": "query: ", "passage": "passage: "},
        "parent_commit": PARENT_COMMIT,
        "parent_report_sha256": frozen.sha256_file(args.parent_report),
        "scorer_sha256": scorer_sha256,
        "code_sha256": m05_code_hashes(),
        "validation_ranking_artifacts": artifacts,
    }
    frozen.write_json(args.selection, selection)
    print(json.dumps({"selected": selection["selected_config"], "gates": gates, "candidates": list(summaries)}, indent=2))


def run_config(args) -> None:
    config = canonical_config(CONFIGS[args.config_name])
    cases, fixture_manifest = frozen.load_fixture(args.output_dir, args.split)
    expected_fixture = frozen.EXPECTED_FIXTURE_SHA256 if args.split == "test" else frozen.EXPECTED_VALIDATION_FIXTURE_SHA256
    if fixture_manifest["fixture_sha256"] != expected_fixture:
        raise ValueError("M05 fixture does not match the frozen contract")
    if args.split == "test" and not args.selection:
        raise ValueError("M05 public test runs require a frozen validation selection")
    selection = None
    if args.selection:
        selection = json.loads(args.selection.read_text(encoding="utf8"))
        if selection.get("experiment") != "M05" or selection.get("split") != "validation" or selection.get("test_accessed") is not False:
            raise ValueError("M05 selection contract mismatch")
        if (
            args.config_name not in selection["candidates"]
            or canonical_config(selection["candidates"][args.config_name]["config"]) != config
            or config_sha256(config) != selection["candidates"][args.config_name]["config_sha256"]
        ):
            raise ValueError("M05 configuration was not frozen on validation")
        verify_validation_evidence(
            selection, args.selection, args.corpus_db, args.scorer_package, args.bootstrap, args.seed
        )
        if frozen.sha256_file(args.representation_manifest) != selection["representation_manifest_sha256"]:
            raise ValueError("M05 representation manifest changed after selection")
    manifest = load_representation_manifest(args.representation_manifest)
    deployment, paths, snapshot, databases = load_or_build_indexes(
        args, cases, fixture_manifest, manifest, (args.config_name,)
    )
    activate_config(deployment, databases, args.config_name)
    report = config_report(args, args.config_name, cases, fixture_manifest, deployment, paths, snapshot, manifest)
    if selection:
        if report["index_artifacts"][args.config_name] != selection["index_artifacts"][args.config_name]:
            raise ValueError(f"M05 {args.config_name} index artifacts changed after selection")
        report["selection_sha256"] = frozen.sha256_file(args.selection)
        report["selection"] = selection
        report["representation_manifest_sha256"] = frozen.sha256_file(args.representation_manifest)
    frozen.write_json(args.output, report)
    print(json.dumps({"output": str(args.output), "config": config, "queries": len(cases)}, indent=2))


def loaded_config_name(loaded_name, selection):
    if loaded_name == "candidate":
        return selection["selected_name"]
    return {
        "off": "parent_translation",
        "sentences_1": "translation_sentences_1",
        "sentences_2": "translation_sentences_2",
    }[loaded_name]


def active_index_artifact_valid(report, selection, loaded_name):
    config_name = loaded_config_name(loaded_name, selection)
    return (
        report.get("index_artifacts", {}).get(config_name)
        == selection.get("index_artifacts", {}).get(config_name)
    )


def source_attribution_valid(report):
    try:
        config = canonical_config(report["config"])
        runs = report["runs"]
        if not isinstance(runs, dict):
            return False
        for run in runs.values():
            if not isinstance(run, dict):
                return False
            refs = run.get("refs")
            fields = run.get("source_fields")
            kinds = run.get("span_kinds")
            offsets = run.get("span_offsets")
            sizes = run.get("span_sentences")
            chunk_ids = run.get("chunk_ids")
            if not all(isinstance(value, list) and len(value) == len(refs) for value in (fields, kinds, offsets, sizes, chunk_ids)):
                return False
            if not all(isources == ["translation"] for isources in fields):
                return False
            if not all(kind in ("whole", "sentence") for kind in kinds):
                return False
            if config["span_mode"] == "whole" and any(kind != "whole" for kind in kinds):
                return False
            if not all(isinstance(doc_id, str) and doc_id for doc_id in chunk_ids):
                return False
            for ref, kind, offset, size, doc_id in zip(refs, kinds, offsets, sizes, chunk_ids):
                if (
                    not isinstance(offset, list) or len(offset) != 2
                    or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in offset)
                    or offset[0] >= offset[1] or isinstance(size, bool) or size not in (1, 2)
                ):
                    return False
                if kind == "whole" and offset[0] != 0:
                    return False
                if kind == "sentence" and size != config["span_sentences"]:
                    return False
                if config["span_mode"] == "sentences":
                    match = sentence_spans.SPAN_ID_RE.fullmatch(doc_id)
                    expected_suffix = "whole" if kind == "whole" else None
                    if (
                        match is None or match["field"] != "translation"
                        or f"{match['surah']}:{match['ayah']}" != ref
                        or (kind == "whole" and match["suffix"] != expected_suffix)
                        or (kind == "sentence" and (match["suffix"] == "whole" or int(match["chunk"]) < 1))
                    ):
                        return False
        return True
    except (AttributeError, KeyError, TypeError, ValueError):
        return False


def _write_query_rankings(path: Path, cases, rankings, report):
    frozen.write_jsonl(path, (
        {
            "id": case["id"], "group_id": case["group_id"], "query": case["query"],
            "ranking": rankings[case["id"]],
            "source_fields": report["runs"][case["id"]]["source_fields"],
            "span_kinds": report["runs"][case["id"]]["span_kinds"],
            "chunk_ids": report["runs"][case["id"]]["chunk_ids"],
            "span_offsets": report["runs"][case["id"]]["span_offsets"],
        }
        for case in cases
    ))


def _slice_summary(selected, scores, bootstrap, seed):
    row = frozen.aggregate(selected, scores, bootstrap, seed)
    row["counts"] = {
        metric: sum(scores[case["id"]][metric] > 0 for case in selected)
        for metric in ("positive_hit@5", "useful_hit@5")
    }
    return {"queries": len(selected), **row}


def _long_document_slice(cases, scores, long_refs, bootstrap, seed):
    selected = [case for case in cases if set(case["qrels"]) & long_refs]
    return _slice_summary(selected, scores, bootstrap, seed)


def build_report(args) -> None:
    cases, fixture_manifest = frozen.load_fixture(args.output_dir, "test")
    if fixture_manifest["fixture_sha256"] != frozen.EXPECTED_FIXTURE_SHA256:
        raise ValueError("M05 test fixture does not match the frozen contract")
    corpus = frozen.load_corpus(args.corpus_db)
    corpus_refs = set(corpus)
    score_group = frozen.scorer_module(args.scorer_package)
    loaded = {}
    for name, path in (
        ("candidate", args.candidate), ("off", args.off_candidate),
        ("sentences_1", args.sentences_1_candidate), ("sentences_2", args.sentences_2_candidate),
    ):
        report, rankings = frozen.load_run(path, cases, corpus_refs, fixture_manifest)
        loaded[name] = {"path": path, "report": report, "rankings": rankings, "scores": frozen.score_run(cases, rankings, score_group)}
    selection = json.loads(args.selection.read_text(encoding="utf8"))
    verify_validation_evidence(
        selection, args.selection, args.corpus_db, args.scorer_package, args.bootstrap, args.seed
    )
    parent_report = json.loads((args.baseline_dir / "report.json").read_text(encoding="utf8"))
    if parent_report.get("experiment") != "M04" or parent_report.get("outcome") != "accepted" or parent_report.get("accepted_run") != "m01":
        raise ValueError("M05 parent is not the accepted M04 infrastructure stack")
    parent_artifact = parent_report["query_ranking_artifacts"]["m04"]
    parent_path = args.baseline_dir / parent_artifact["file"]
    if frozen.sha256_file(parent_path) != parent_artifact["sha256"]:
        raise ValueError("M04 committed ranking hash mismatch")
    parent_rows = frozen.read_jsonl(parent_path)
    parent_rankings = {row["id"]: row["ranking"] for row in parent_rows}
    parent_scores = frozen.score_run(cases, parent_rankings, score_group)
    browser_report = json.loads((args.browser_baseline_dir / "report.json").read_text(encoding="utf8"))
    browser_rankings, _ = frozen.load_committed_rankings("browser", args.browser_baseline_dir, browser_report, cases, corpus_refs)
    browser_scores = frozen.score_run(cases, browser_rankings, score_group)
    summaries = {
        name: _headline_summary(value["report"], cases, value["rankings"], value["scores"], args.bootstrap, args.seed)
        for name, value in loaded.items()
    }
    candidate_summary = summaries["candidate"]
    comparisons = {}
    for candidate_name, baseline_name, baseline_scores, baseline_rankings in (
        ("candidate", "m04", parent_scores, parent_rankings),
        ("candidate", "browser", browser_scores, browser_rankings),
        ("off", "m04", parent_scores, parent_rankings),
        ("sentences_1", "m04", parent_scores, parent_rankings),
        ("sentences_2", "m04", parent_scores, parent_rankings),
        ("sentences_1", "off", loaded["off"]["scores"], loaded["off"]["rankings"]),
        ("sentences_2", "off", loaded["off"]["scores"], loaded["off"]["rankings"]),
    ):
        comparisons[f"m05_{candidate_name}_vs_{baseline_name}"] = m03_tafsir._comparison(
            candidate_name, cases, baseline_scores, loaded[candidate_name]["scores"],
            baseline_rankings, loaded[candidate_name]["rankings"], args.bootstrap, args.seed,
        )
    candidate_counts = frozen.headline_test_counts(candidate_summary)
    parent_counts = frozen.headline_test_counts(parent_report["runs"]["m04"])
    test_gate = all(candidate_counts[key] >= parent_counts[key] for key in parent_counts)
    budget = parent_report["runtime_budgets"]["runs"]["dense"]
    measurements = {
        "p95_query_ms": candidate_summary["timings_ms"]["p95"],
        "index_bytes": candidate_summary["index_bytes"],
        "peak_rss_bytes": candidate_summary["peak_rss_bytes"],
    }
    runtime_gate = all(measurements[key] <= budget[f"{key}_max"] for key in measurements)
    selected_config = canonical_config(selection["selected_config"])
    candidate_config = canonical_config(loaded["candidate"]["report"]["config"])
    selection_hash = frozen.sha256_file(args.selection)
    selection_contract = (
        selection.get("experiment") == "M05" and selection.get("split") == "validation"
        and selection.get("test_accessed") is False
        and selection["dataset"]["fixture_sha256"] == frozen.EXPECTED_VALIDATION_FIXTURE_SHA256
        and selection["determinism_check"]["passed"]
        and selection.get("parent_commit") == PARENT_COMMIT
        and selection.get("selected_name") in selection.get("candidates", {})
        and config_sha256(selection.get("selected_config", {})) == selection.get("selected_config_sha256")
        and selection.get("representation_manifest_sha256") == frozen.sha256_file(args.representation_manifest)
        and selection.get("scorer_sha256") == frozen.text_sha256(args.scorer_package / "evaluate_search.py")
        and selection.get("parent_report_sha256") == frozen.sha256_file(args.baseline_dir / "report.json")
    )
    try:
        verify_frozen_code(selection)
        code_unchanged = True
    except ValueError:
        code_unchanged = False
    parent_provenance = parent_report["runs"]["m04"]["provenance"]
    fixed_keys = (
        "model", "model_revision", "model_files", "dtype", "distance", "field", "display_field",
        "corpus", "query_prefix", "passage_prefix", "retrieval", "aggregation",
    )
    fixed_match = all(loaded["candidate"]["report"].get(key) == parent_provenance.get(key) for key in fixed_keys)
    manifest_match = loaded["candidate"]["report"].get("representation_manifest_sha256") == frozen.sha256_file(args.representation_manifest)
    expected_configs = {
        "candidate": selected_config,
        "off": CONFIGS["parent_translation"],
        "sentences_1": CONFIGS["translation_sentences_1"],
        "sentences_2": CONFIGS["translation_sentences_2"],
    }
    contract_flags = {}
    for name, value in loaded.items():
        report = value["report"]
        contract_flags[name] = (
            canonical_config(report["config"]) == expected_configs[name]
            and report.get("selection_sha256") == selection_hash
            and report.get("selection") == selection
            and report.get("representation_manifest_sha256") == selection["representation_manifest_sha256"]
            and source_attribution_valid(report)
            and active_index_artifact_valid(report, selection, name)
        )
    off_exact = sum(loaded["off"]["rankings"][case["id"]] == parent_rankings[case["id"]] for case in cases)
    off_parity = off_exact == len(cases) and contract_flags["off"]
    validation_gate_passed = not selection["selected_changed"] or selection["validation_gate_passed"]
    checks = json.loads(args.checks.read_text(encoding="utf8"))
    accepted = all((
        candidate_config == selected_config, selection_contract, code_unchanged, fixed_match,
        manifest_match, all(contract_flags.values()), test_gate, runtime_gate, off_parity,
        validation_gate_passed, checks.get("status") == "passed",
    ))
    if accepted and selection["selected_changed"]:
        candidate_summary["leaderboard_label"] = f"M05 sentences-{selected_config['span_sentences']} (accepted)"
    elif not accepted:
        candidate_summary["leaderboard_label"] = "M05 translation (rejected; parent retained)"
    else:
        candidate_summary["leaderboard_label"] = "M05 translation (parent retained)"
    long_refs = set(selection["representation_manifest"]["outputs"]["sentences_1"]["tokens"]["long_document_ayahs"])
    long_slices = {
        name: _long_document_slice(cases, value["scores"], long_refs, args.bootstrap, args.seed)
        for name, value in {**loaded, "m04": {"scores": parent_scores}}.items()
    }
    index_cost = {
        name: {
            "bytes": summary["index_bytes"],
            "delta_vs_parent_bytes": summary["index_bytes"] - summaries["off"]["index_bytes"],
            "within_fixed_budget": summary["index_bytes"] <= budget["index_bytes_max"],
        }
        for name, summary in {"parent": summaries["off"], **{name: summaries[name] for name in ("sentences_1", "sentences_2")}}.items()
    }
    report = {
        "profile": "pr2_frozen_v1",
        "experiment": "M05",
        "outcome": "accepted" if accepted else "rejected",
        "accepted_run": "m05" if accepted and selection["selected_changed"] else "m01",
        "parent_commit": PARENT_COMMIT,
        "contribution": "Optional deterministic one- and two-sentence spans with exact source offsets, retained whole documents, bounded span tokens, stable IDs, per-ayah max-score aggregation, and retained-translation display.",
        "dataset": fixture_manifest,
        "selection": selection,
        "selection_sha256": selection_hash,
        "representation_manifest": selection["representation_manifest"],
        "representation_manifest_sha256": selection["representation_manifest_sha256"],
        "parent_report_sha256": frozen.sha256_file(args.baseline_dir / "report.json"),
        "scorer_sha256": frozen.text_sha256(args.scorer_package / "evaluate_search.py"),
        "code_sha256": m05_code_hashes(),
        "runs": {**parent_report["runs"], "m05": candidate_summary},
        "diagnostic_runs": {"off": summaries["off"], "sentences_1": summaries["sentences_1"], "sentences_2": summaries["sentences_2"]},
        "comparisons": comparisons,
        "on_off_ablation": {
            "off": CONFIGS["parent_translation"], "on": selected_config,
            "sentences_1": CONFIGS["translation_sentences_1"], "sentences_2": CONFIGS["translation_sentences_2"],
            "off_exact_parent_queries": off_exact,
            "candidate_exact_off_queries": sum(loaded["candidate"]["rankings"][case["id"]] == loaded["off"]["rankings"][case["id"]] for case in cases),
            "candidate_exact_parent_queries": sum(loaded["candidate"]["rankings"][case["id"]] == parent_rankings[case["id"]] for case in cases),
            "off_headline_counts": frozen.headline_test_counts(summaries["off"]),
            "off_timings_ms": summaries["off"]["timings_ms"],
            "off_index_bytes": summaries["off"]["index_bytes"],
        },
        "document_slices": {
            "definition": f"Queries with at least one qrel ayah whose whole translation exceeds {LONG_DOCUMENT_TOKENS} E5 tokens.",
            "long_document": long_slices,
            "compound": {name: summary["headline"]["compound"] for name, summary in {**summaries, "m04": parent_report["runs"]["m04"]}.items()},
        },
        "index_cost": index_cost,
        "runtime_budgets": parent_report["runtime_budgets"],
        "runtime_budget_check": {"budget": budget, "measurements": measurements, "passed": runtime_gate},
        "acceptance": {
            "accepted": accepted,
            "config_matches_frozen_selection": candidate_config == selected_config,
            "validation_only_selection_contract": selection_contract,
            "frozen_code_unchanged": code_unchanged,
            "fixed_model_corpus_aggregation_contract_matches": fixed_match,
            "sentence_span_manifest_matches": manifest_match,
            "frozen_run_contracts": contract_flags,
            "source_attribution_valid": all(source_attribution_valid(value["report"]) for value in loaded.values()),
            "validation_gate_passed": validation_gate_passed,
            "test_gate_passed": test_gate,
            "runtime_budget_passed": runtime_gate,
            "similarity_rollback_parity": off_parity,
            "targeted_checks_passed": checks.get("status") == "passed",
            "parent_headline_counts": parent_counts,
            "candidate_headline_counts": candidate_counts,
        },
        "targeted_checks": checks,
        "limitations": [
            *fixture_manifest["notes"],
            "Whole documents are retained even when they exceed 512 tokens; every added sentence span is bounded at 512 tokens.",
            "Single-sentence verses can have a sentence span identical to their retained whole document; per-ayah max-score collapse prevents duplicate votes.",
            "Sentence boundaries are conservative and deterministic, not a claim of linguistic sentence parsing.",
            "Optional span indexes are measured separately and may exceed the fixed dense budget; only the selected active path is subject to promotion runtime gates.",
            "The English IslamQA score does not establish Arabic/Bengali semantic quality.",
            "Runtime budgets remain fixed from M00.",
        ],
    }
    ranking_outputs = {
        "m05": (args.output_dir / "m05_query_rankings.jsonl", "candidate"),
        "m05_off": (args.output_dir / "m05_off_query_rankings.jsonl", "off"),
        "m05_sentences_1": (args.output_dir / "m05_sentences_1_query_rankings.jsonl", "sentences_1"),
        "m05_sentences_2": (args.output_dir / "m05_sentences_2_query_rankings.jsonl", "sentences_2"),
    }
    report["query_ranking_artifacts"] = {}
    report["source_ranking_artifacts"] = {}
    for name, (path, loaded_name) in ranking_outputs.items():
        value = loaded[loaded_name]
        _write_query_rankings(path, cases, value["rankings"], value["report"])
        report["query_ranking_artifacts"][name] = {"file": path.name, "sha256": frozen.sha256_file(path), "bytes": path.stat().st_size}
        report["source_ranking_artifacts"][name] = {"file": value["path"].name, "sha256": frozen.sha256_file(value["path"]), "bytes": value["path"].stat().st_size}
    table = frozen.markdown_leaderboard(report)
    (args.output_dir / "leaderboard.md").write_text(table + "\n", encoding="utf8")
    frozen.write_json(args.output_dir / "report.json", report)
    print(table)
    print(json.dumps(report["acceptance"], indent=2))


def add_parsers(subparsers) -> None:
    export_parser = subparsers.add_parser("m05-export", help="Export audited whole and sentence-span representations")
    export_parser.add_argument("--corpus-db", type=Path, default=frozen.DEFAULT_AAYATUN_REPO / "db" / "en_quran.db")
    export_parser.add_argument("--output-dir", type=Path, default=DEFAULT_M05_VALIDATION)
    export_parser.set_defaults(func=export_spans)

    sweep_parser = subparsers.add_parser("m05-sweep", help="Select M05 span size on validation only")
    sweep_parser.add_argument("--corpus-db", type=Path, default=frozen.DEFAULT_AAYATUN_REPO / "db" / "en_quran.db")
    sweep_parser.add_argument("--scorer-package", type=Path, default=frozen.DEFAULT_SCORER_PACKAGE)
    sweep_parser.add_argument("--output-dir", type=Path, default=DEFAULT_M05_VALIDATION)
    sweep_parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_M05_VALIDATION / "runtime")
    sweep_parser.add_argument("--spans-output-1", type=Path, default=DEFAULT_M05_VALIDATION / "runtime" / "translation_sentence_spans_1.jsonl")
    sweep_parser.add_argument("--spans-output-2", type=Path, default=DEFAULT_M05_VALIDATION / "runtime" / "translation_sentence_spans_2.jsonl")
    sweep_parser.add_argument("--representation-manifest", type=Path, default=DEFAULT_M05_VALIDATION / "representation_manifest.json")
    sweep_parser.add_argument("--selection", type=Path, default=DEFAULT_M05_VALIDATION / "selection.json")
    sweep_parser.add_argument("--parent-report", type=Path, default=DEFAULT_M04_TEST / "report.json")
    sweep_parser.add_argument("--bootstrap", type=int, default=2000)
    sweep_parser.add_argument("--seed", type=int, default=4213)
    sweep_parser.set_defaults(func=run_sweep)

    config_parser = subparsers.add_parser("m05-config", help="Run one frozen M05 configuration")
    config_parser.add_argument("--config-name", choices=tuple(CONFIGS), required=True)
    config_parser.add_argument("--corpus-db", type=Path, default=frozen.DEFAULT_AAYATUN_REPO / "db" / "en_quran.db")
    config_parser.add_argument("--scorer-package", type=Path, default=frozen.DEFAULT_SCORER_PACKAGE)
    config_parser.add_argument("--output-dir", type=Path, default=DEFAULT_M05_TEST)
    config_parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_M05_VALIDATION / "runtime")
    config_parser.add_argument("--spans-output-1", type=Path, default=DEFAULT_M05_VALIDATION / "runtime" / "translation_sentence_spans_1.jsonl")
    config_parser.add_argument("--spans-output-2", type=Path, default=DEFAULT_M05_VALIDATION / "runtime" / "translation_sentence_spans_2.jsonl")
    config_parser.add_argument("--representation-manifest", type=Path, default=DEFAULT_M05_VALIDATION / "representation_manifest.json")
    config_parser.add_argument("--output", type=Path, required=True)
    config_parser.add_argument("--split", choices=("test", "validation"), default="test")
    config_parser.add_argument("--selection", type=Path)
    config_parser.add_argument("--bootstrap", type=int, default=2000)
    config_parser.add_argument("--seed", type=int, default=4213)
    config_parser.set_defaults(func=run_config)

    report_parser = subparsers.add_parser("m05-report", help="Score frozen M05 and append its leaderboard column")
    report_parser.add_argument("--corpus-db", type=Path, default=frozen.DEFAULT_AAYATUN_REPO / "db" / "en_quran.db")
    report_parser.add_argument("--scorer-package", type=Path, default=frozen.DEFAULT_SCORER_PACKAGE)
    report_parser.add_argument("--output-dir", type=Path, default=DEFAULT_M05_TEST)
    report_parser.add_argument("--candidate", type=Path, default=DEFAULT_M05_TEST / "m05_rankings.json")
    report_parser.add_argument("--off-candidate", type=Path, default=DEFAULT_M05_TEST / "m05_off_rankings.json")
    report_parser.add_argument("--sentences-1-candidate", type=Path, default=DEFAULT_M05_TEST / "m05_sentences_1_rankings.json")
    report_parser.add_argument("--sentences-2-candidate", type=Path, default=DEFAULT_M05_TEST / "m05_sentences_2_rankings.json")
    report_parser.add_argument("--selection", type=Path, default=DEFAULT_M05_VALIDATION / "selection.json")
    report_parser.add_argument("--representation-manifest", type=Path, default=DEFAULT_M05_VALIDATION / "representation_manifest.json")
    report_parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_M04_TEST)
    report_parser.add_argument("--browser-baseline-dir", type=Path, default=frozen.DEFAULT_OUTPUT)
    report_parser.add_argument("--checks", type=Path, default=DEFAULT_M05_TEST / "checks.json")
    report_parser.add_argument("--bootstrap", type=int, default=2000)
    report_parser.add_argument("--seed", type=int, default=4213)
    report_parser.set_defaults(func=build_report)
