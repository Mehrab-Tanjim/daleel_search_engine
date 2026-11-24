from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any
from unittest import mock

try:
    from . import islamqa_frozen as frozen
    from . import representations
except ImportError:
    import islamqa_frozen as frozen
    import representations


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_M03_VALIDATION = HERE / "results" / "m03_tafsir" / "validation"
DEFAULT_M03_TEST = HERE / "results" / "m03_tafsir" / "test"
DEFAULT_M02_VALIDATION = HERE / "results" / "m02_ai" / "validation"
DEFAULT_M02_TEST = HERE / "results" / "m02_ai" / "test"
SCOPE_FIELDS = {
    "translation": ("translation",),
    "ai": ("ai",),
    "translation_ai": ("translation", "ai"),
    "en_tafsir": ("en_tafsir",),
    "translation_en_tafsir": ("translation", "en_tafsir"),
    "ai_en_tafsir": ("ai", "en_tafsir"),
    "translation_ai_en_tafsir": ("translation", "ai", "en_tafsir"),
}
CONFIGS = {
    "parent_translation": {"semantic_ai": False, "semantic_en_tafsir": False, "representation_scope": "translation"},
    "tafsir_only_diagnostic": {"semantic_ai": False, "semantic_en_tafsir": True, "representation_scope": "en_tafsir"},
    "parent_tafsir": {"semantic_ai": False, "semantic_en_tafsir": True, "representation_scope": "translation_en_tafsir"},
    "parent_ai": {"semantic_ai": True, "semantic_en_tafsir": False, "representation_scope": "translation_ai"},
    "ai_tafsir_only_diagnostic": {"semantic_ai": True, "semantic_en_tafsir": True, "representation_scope": "ai_en_tafsir"},
    "parent_ai_tafsir": {"semantic_ai": True, "semantic_en_tafsir": True, "representation_scope": "translation_ai_en_tafsir"},
}


def canonical_config(config: dict[str, Any]) -> dict[str, Any]:
    semantic_ai = config["semantic_ai"]
    semantic_en_tafsir = config["semantic_en_tafsir"]
    if not isinstance(semantic_ai, bool):
        raise ValueError("semantic_ai must be a boolean")
    if not isinstance(semantic_en_tafsir, bool):
        raise ValueError("semantic_en_tafsir must be a boolean")
    scope = config["representation_scope"]
    if scope not in SCOPE_FIELDS:
        raise ValueError("Unknown M03 representation scope")
    fields = SCOPE_FIELDS[scope]
    if "ai" in fields and not semantic_ai:
        raise ValueError("AI scopes require semantic_ai=true")
    if "en_tafsir" in fields and not semantic_en_tafsir:
        raise ValueError("Tafsir scopes require semantic_en_tafsir=true")
    return {
        "semantic_ai": semantic_ai,
        "semantic_en_tafsir": semantic_en_tafsir,
        "representation_scope": scope,
    }


def config_sha256(config: dict[str, Any]) -> str:
    payload = json.dumps(canonical_config(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf8")).hexdigest()


def export_tafsir(args) -> None:
    from transformers import AutoTokenizer

    corpus = frozen.load_corpus(args.corpus_db)
    if len(corpus) != 6236 or frozen.corpus_hash(corpus) != frozen.EXPECTED_CORPUS_SHA256:
        raise ValueError("Corpus does not match the frozen fixture")
    train_records = json.loads(args.train_sources.read_text(encoding="utf8"))
    validation_questions = frozen.read_jsonl(args.validation_questions)
    source_records = [
        {"source_file": args.train_sources.name, "question": row.get("question"), "answer": row.get("answer")}
        for row in train_records
    ]
    source_records.extend(
        {"source_file": "validation_questions.jsonl", "question": query["text"], "answer": ""}
        for row in validation_questions
        for query in row["queries"]
    )
    source_audit = args.output_dir / "runtime" / "train_validation_sources.json"
    frozen.write_json(source_audit, source_records)
    tokenizer = AutoTokenizer.from_pretrained(frozen.model_snapshot(), local_files_only=True)
    manifest = representations.export_tafsir_representation(
        args.tafsir_db,
        corpus,
        args.output,
        [source_audit],
        tokenizer=tokenizer,
        max_tokens=512,
    )
    for section, key in (("token_truncation", "truncated_ayahs_file"), ("repeated_commentary", "sidecar_file")):
        value = manifest.get(section) or {}
        if section == "repeated_commentary":
            value = value["exported"]
        if value.get(key):
            value[key] = Path(os.path.relpath(args.output.parent / value[key], args.output_dir)).as_posix()
    manifest["selection_sources"] = {
        "dataset_commit": frozen.DATASET_COMMIT,
        "train_sources": {"file": args.train_sources.name, "sha256": frozen.sha256_file(args.train_sources), "records": len(train_records)},
        "validation_questions": {
            "file": args.validation_questions.name,
            "sha256": frozen.sha256_file(args.validation_questions),
            "groups": len(validation_questions),
            "wordings": sum(len(row["queries"]) for row in validation_questions),
        },
        "combined_audit_file": Path(os.path.relpath(source_audit, args.output_dir)).as_posix(),
        "combined_audit_sha256": frozen.sha256_file(source_audit),
        "test_sources_used": False,
        "limitation": "The legacy train file has historical validation/test exposure; no public test source was used for M03 export or selection.",
    }
    frozen.write_json(args.output_dir / "representation_manifest.json", manifest)
    print(json.dumps({
        "output": str(args.output),
        "manifest": str(args.output_dir / "representation_manifest.json"),
        "counts": manifest["counts"],
        "truncated": manifest["token_truncation"]["truncated_documents"],
        "overlaps": manifest["provenance_audit"]["overlap_count"],
        "repeated_spans": manifest["repeated_commentary"]["exported"]["unique_repeated_spans"],
    }, indent=2))


def _verify_sidecar(manifest: dict[str, Any], path: Path, section: str, file_key: str, hash_key: str) -> None:
    value = manifest.get(section) or {}
    if section == "repeated_commentary":
        value = value["exported"]
    if value.get(file_key):
        if frozen.sha256_file(path.parent / value[file_key]) != value[hash_key]:
            raise ValueError(f"M03 {section} sidecar hash mismatch")


def load_representation_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf8"))
    if manifest.get("experiment") != "M03" or manifest.get("representation") != "en_tafsir":
        raise ValueError("Invalid M03 representation manifest")
    if manifest["source"]["source_db_sha256"] != representations.TAFSIR_DB_SHA256:
        raise ValueError("M03 tafsir source hash changed")
    if manifest["corpus"]["sha256"] != frozen.EXPECTED_CORPUS_SHA256:
        raise ValueError("M03 representation corpus hash changed")
    if manifest.get("selection_sources", {}).get("test_sources_used") is not False:
        raise ValueError("M03 export must not use test sources")
    _verify_sidecar(manifest, path, "token_truncation", "truncated_ayahs_file", "truncated_ayahs_sha256")
    _verify_sidecar(manifest, path, "repeated_commentary", "sidecar_file", "sidecar_sha256")
    return manifest


def load_ai_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf8"))
    if manifest.get("experiment") != "M02" or manifest.get("representation") != "ai":
        raise ValueError("Invalid inherited M02 AI manifest")
    if manifest["source"]["source_db_sha256"] != representations.AI_DB_SHA256:
        raise ValueError("Inherited AI source hash changed")
    if manifest["corpus"]["sha256"] != frozen.EXPECTED_CORPUS_SHA256:
        raise ValueError("Inherited AI corpus hash changed")
    truncation = manifest.get("token_truncation") or {}
    if truncation.get("truncated_ayahs_file"):
        if frozen.sha256_file(path.parent / truncation["truncated_ayahs_file"]) != truncation["truncated_ayahs_sha256"]:
            raise ValueError("Inherited AI truncated-ayah sidecar hash mismatch")
    return manifest


def _load_optional_index(deployment, path: Path, output: Path, manifest: dict[str, Any], field: str, source: str, corpus_refs: set[str]) -> Path:
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document

    rows = frozen.read_jsonl(output)
    if frozen.sha256_file(output) != manifest["output"]["sha256"]:
        raise ValueError(f"{field} export hash mismatch")
    refs = [row["ayah"] for row in rows]
    if len(refs) != len(set(refs)) or set(refs) - corpus_refs:
        raise ValueError(f"{field} export has duplicate or unknown ayat")
    if any(row.get("field") != field or row.get("source") != source for row in rows):
        raise ValueError(f"{field} source attribution mismatch")
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = {f"{field}:{row['ayah']}": row for row in rows}
    if not (path / "index.faiss").exists():
        documents = [
            Document(
                page_content="passage: " + row["text"],
                metadata={
                    "surah": int(row["ayah"].split(":")[0]),
                    "ayah": int(row["ayah"].split(":")[1]),
                    "field": field,
                    "source": source,
                    "chunk": 0,
                },
                id=doc_id,
            )
            for doc_id, row in sorted(expected.items(), key=lambda item: representations.ayah_sort_key(item[1]["ayah"]))
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
    setattr(deployment, f"{field}_db", database)
    setattr(deployment, f"{field}_db_docstore_id_to_index", dict(
        zip(database.index_to_docstore_id.values(), database.index_to_docstore_id.keys())
    ))
    if database.index.ntotal != len(rows):
        raise ValueError(f"{field} index size mismatch")
    if set(database.index_to_docstore_id.values()) != set(expected):
        raise ValueError(f"{field} index document IDs mismatch")
    for doc_id, row in expected.items():
        document = database.docstore.search(doc_id)
        surah, ayah = (int(part) for part in row["ayah"].split(":"))
        expected_metadata = {"surah": surah, "ayah": ayah, "field": field, "source": source, "chunk": 0}
        if document is None or document.page_content != "passage: " + row["text"] or document.metadata != expected_metadata:
            raise ValueError(f"{field} index content mismatch: {row['ayah']}")
    return path


def load_or_build_indexes(args, cases, fixture_manifest, tafsir_manifest, ai_manifest, load_ai=False, load_tafsir=False):
    with mock.patch.dict(os.environ, {"SEMANTIC_AI": "false", "SEMANTIC_EN_TAFSIR": "false"}, clear=False):
        deployment, translation_path, snapshot = frozen.load_or_build_dense_index(args, cases, fixture_manifest)
    corpus_refs = set(frozen.load_corpus(args.corpus_db))
    paths = {
        "translation": translation_path,
        "ai": args.runtime_dir / "e5-small" / "quran-ai",
        "en_tafsir": args.runtime_dir / "e5-small" / "quran-en-tafsir",
    }
    if load_ai:
        _load_optional_index(deployment, paths["ai"], args.ai_output, ai_manifest, "ai", "aayatun_ai_generated", corpus_refs)
    if load_tafsir:
        _load_optional_index(deployment, paths["en_tafsir"], args.tafsir_output, tafsir_manifest, "en_tafsir", "ibn_kathir_english", corpus_refs)
    return deployment, paths, snapshot


def index_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def index_artifacts(path: Path) -> dict[str, dict[str, Any]]:
    return {
        item.relative_to(path).as_posix(): {"sha256": frozen.sha256_file(item), "bytes": item.stat().st_size}
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }


def execute_config(deployment, cases, config):
    config = canonical_config(config)
    runs = {}
    for index, case in enumerate(cases):
        started = time.perf_counter()
        results = deployment.search(
            "best_match",
            "query: " + case["query"],
            k=50,
            retrieval_mode="similarity",
            fetch_k=50,
            mmr_lambda=1.0,
            semantic_ai=config["semantic_ai"],
            semantic_en_tafsir=config["semantic_en_tafsir"],
            representation_scope=config["representation_scope"],
        )
        elapsed = round((time.perf_counter() - started) * 1000, 3)
        refs = [f"{doc.metadata['surah']}:{doc.metadata['ayah']}" for doc, _ in results]
        fields = [doc.metadata["field"] for doc, _ in results]
        if len(refs) != len(set(refs)) or len(refs) > 50:
            raise ValueError(f"M03 duplicate or over-depth result: {case['id']}")
        if set(fields) - set(SCOPE_FIELDS[config["representation_scope"]]):
            raise ValueError(f"M03 source-field leakage: {case['id']}")
        runs[case["id"]] = {"refs": refs, "source_fields": [[field] for field in fields], "ms": elapsed}
        if index % 50 == 0:
            print(f"M03 {config['representation_scope']}: {index}/{len(cases)}", flush=True)
    return {"config": config, "result_k": 50, "runs": runs}


def config_report(args, cases, fixture_manifest, deployment, paths, snapshot, tafsir_manifest, ai_manifest, config):
    execution = execute_config(deployment, cases, config)
    fields = SCOPE_FIELDS[execution["config"]["representation_scope"]]
    active_paths = [paths[field] for field in fields]
    report = {
        "engine": "M03 offline multilingual E5-small float32 FAISS with optional English tafsir",
        "profile": "pr2_frozen_v1",
        "experiment": "M03",
        "model": frozen.E5_MODEL,
        "model_revision": frozen.E5_REVISION,
        "model_files": frozen.dense_model_files(snapshot),
        "dtype": "float32",
        "distance": "max_inner_product",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
        "field": "translation",
        "display_field": "translation",
        "source_attribution": "per-result winning representation",
        "fixture_sha256": fixture_manifest["fixture_sha256"],
        "corpus": {"verses": 6236, "sha256": fixture_manifest["corpus"]["sha256"]},
        "retrieval": {"mode": "similarity", "fetch_k": 50, "mmr_lambda": 1.0},
        "aggregation": "Fetch 50 per active representation; keep one maximum-scored result per ayah; ties prefer translation, AI, then English tafsir, followed by numeric ayah ID.",
        "index_bytes": sum(index_bytes(path) for path in active_paths),
        "index_artifacts": {field: index_artifacts(path) for field, path in paths.items()},
        "active_representations": [path.name for path in active_paths],
        "peak_rss_bytes": frozen.peak_rss_bytes(),
        "en_tafsir_representation_manifest_sha256": frozen.sha256_file(args.representation_manifest),
        "ai_representation_manifest_sha256": frozen.sha256_file(args.ai_representation_manifest),
        **execution,
    }
    report["config_sha256"] = config_sha256(report["config"])
    return report


def m03_code_hashes():
    return {
        "evaluation/islamqa_frozen.py": frozen.sha256_file(HERE / "islamqa_frozen.py"),
        "evaluation/m03_tafsir.py": frozen.sha256_file(HERE / "m03_tafsir.py"),
        "evaluation/representations.py": frozen.sha256_file(HERE / "representations.py"),
        "evaluation/test_tafsir_representation.py": frozen.sha256_file(HERE / "test_tafsir_representation.py"),
        "evaluation/test_ai_representation.py": frozen.sha256_file(HERE / "test_ai_representation.py"),
        "evaluation/test_retrieval_config.py": frozen.sha256_file(HERE / "test_retrieval_config.py"),
        "evaluation/test_islamqa_frozen.py": frozen.sha256_file(HERE / "test_islamqa_frozen.py"),
        "utils.py": frozen.sha256_file(ROOT / "utils.py"),
        "build_vector_database.py": frozen.sha256_file(ROOT / "build_vector_database.py"),
    }


def verify_frozen_code(selection):
    for relative, digest in selection["code_sha256"].items():
        if frozen.sha256_file(ROOT / relative) != digest:
            raise ValueError(f"Frozen M03 code changed after selection: {relative}")


def _paired_metrics(cases, baseline, candidate, bootstrap, seed):
    useful = [case for case in cases if any(grade >= 2 for grade in case["qrels"].values())]
    return {
        metric: frozen.paired_details(
            useful if metric == "useful_hit@5" else cases,
            baseline, candidate, metric, bootstrap, seed,
        )
        for metric in ("positive_hit@5", "useful_hit@5", "ndcg@10")
    }


def _interaction_details(cases, parent, tafsir, ai, joint, metric, bootstrap, seed):
    selected = [case for case in cases if any(grade >= 2 for grade in case["qrels"].values())] if metric == "useful_hit@5" else cases
    differences = {
        case["id"]: (joint[case["id"]][metric] - ai[case["id"]][metric]) - (tafsir[case["id"]][metric] - parent[case["id"]][metric])
        for case in selected
    }
    wins = [case for case in selected if differences[case["id"]] > 1e-12]
    losses = [case for case in selected if differences[case["id"]] < -1e-12]
    ties = [case for case in selected if abs(differences[case["id"]]) <= 1e-12]
    return {
        "metric": metric,
        "definition": "(joint AI+tafsir minus parent+AI) minus (parent+tafsir minus parent)",
        "wins": len(wins),
        "losses": len(losses),
        "ties": len(ties),
        "difference_in_differences": round(sum(differences.values()) / len(differences), 6),
        "difference_ci95": frozen.cluster_interval(selected, differences, bootstrap, seed),
        "gained_cases": [{"id": case["id"], "group_id": case["group_id"], "query": case["query"]} for case in wins],
        "lost_cases": [{"id": case["id"], "group_id": case["group_id"], "query": case["query"]} for case in losses],
    }


def _headline_summary(report, cases, rankings, scores, bootstrap, seed):
    summary = frozen.candidate_headline_summary(report, cases, rankings, scores, bootstrap, seed)
    summary["config"] = report["config"]
    summary["config_sha256"] = report["config_sha256"]
    summary.get("provenance", {}).pop("selection", None)
    return summary


def run_sweep(args) -> None:
    cases, fixture_manifest = frozen.load_fixture(args.output_dir, "validation")
    corpus = frozen.load_corpus(args.corpus_db)
    score_group = frozen.scorer_module(args.scorer_package)
    tafsir_manifest = load_representation_manifest(args.representation_manifest)
    ai_manifest = load_ai_manifest(args.ai_representation_manifest)
    deployment, paths, snapshot = load_or_build_indexes(
        args, cases, fixture_manifest, tafsir_manifest, ai_manifest, load_ai=True, load_tafsir=True
    )
    summaries = {}
    scores_by_name = {}
    artifacts = {}
    for name, config in CONFIGS.items():
        report = config_report(args, cases, fixture_manifest, deployment, paths, snapshot, tafsir_manifest, ai_manifest, config)
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
    candidate = summaries["parent_tafsir"]
    selected = candidate if frozen.validation_gate(parent, candidate) else parent
    marginal_effects = {
        name: _paired_metrics(cases, scores_by_name["parent_translation"], scores, args.bootstrap, args.seed)
        for name, scores in scores_by_name.items()
    }
    incremental_effects = {
        "tafsir_without_ai": _paired_metrics(cases, scores_by_name["parent_translation"], scores_by_name["parent_tafsir"], args.bootstrap, args.seed),
        "tafsir_with_ai": _paired_metrics(cases, scores_by_name["parent_ai"], scores_by_name["parent_ai_tafsir"], args.bootstrap, args.seed),
        "ai_without_tafsir": _paired_metrics(cases, scores_by_name["parent_translation"], scores_by_name["parent_ai"], args.bootstrap, args.seed),
        "ai_with_tafsir": _paired_metrics(cases, scores_by_name["parent_tafsir"], scores_by_name["parent_ai_tafsir"], args.bootstrap, args.seed),
    }
    interaction = {
        metric: _interaction_details(
            cases, scores_by_name["parent_translation"], scores_by_name["parent_tafsir"],
            scores_by_name["parent_ai"], scores_by_name["parent_ai_tafsir"], metric, args.bootstrap, args.seed,
        )
        for metric in ("positive_hit@5", "useful_hit@5", "ndcg@10")
    }
    repeat = execute_config(deployment, cases, selected["config"])
    selected_report = json.loads((args.output_dir / "validation_runs" / f"{selected['name']}.json").read_text(encoding="utf8"))
    exact = sum(
        repeat["runs"][case["id"]]["refs"] == selected_report["runs"][case["id"]]["refs"]
        and repeat["runs"][case["id"]]["source_fields"] == selected_report["runs"][case["id"]]["source_fields"]
        for case in cases
    )
    if exact != len(cases):
        raise ValueError("M03 selected configuration is not deterministic")
    selection = {
        "profile": "pr2_frozen_v1",
        "experiment": "M03",
        "split": "validation",
        "dataset": fixture_manifest,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_accessed": False,
        "selection_policy": "Only parent+English tafsir with AI off is promotion-eligible. Require higher validation useful hit@5 with no validation nDCG@10 decrease; tafsir-only and all AI combinations are diagnostic.",
        "candidates": summaries,
        "marginal_effects": marginal_effects,
        "incremental_effects": incremental_effects,
        "joint_interaction": interaction,
        "selected_name": selected["name"],
        "selected_config": selected["config"],
        "selected_config_sha256": selected["config_sha256"],
        "selected_changed": selected["config_sha256"] != parent["config_sha256"],
        "validation_gate_passed": frozen.validation_gate(parent, candidate),
        "determinism_check": {"queries": len(cases), "exact_queries": exact, "passed": exact == len(cases)},
        "representation_manifest": tafsir_manifest,
        "representation_manifest_sha256": frozen.sha256_file(args.representation_manifest),
        "ai_representation_manifest_sha256": frozen.sha256_file(args.ai_representation_manifest),
        "index_artifacts": {field: index_artifacts(path) for field, path in paths.items()},
        "model": {"name": frozen.E5_MODEL, "revision": frozen.E5_REVISION, "files": frozen.dense_model_files(snapshot)},
        "prefixes": {"query": "query: ", "passage": "passage: "},
        "code_sha256": m03_code_hashes(),
        "validation_ranking_artifacts": artifacts,
    }
    frozen.write_json(args.selection, selection)
    print(json.dumps({"selected": selection["selected_config"], "gate": selection["validation_gate_passed"], "candidates": list(summaries)}, indent=2))


def run_config(args) -> None:
    config = canonical_config(CONFIGS[args.config_name])
    cases, fixture_manifest = frozen.load_fixture(args.output_dir, args.split)
    if args.selection:
        selection = json.loads(args.selection.read_text(encoding="utf8"))
        if selection.get("experiment") != "M03" or selection.get("split") != "validation":
            raise ValueError("M03 selection contract mismatch")
        if args.config_name not in selection["candidates"] or canonical_config(selection["candidates"][args.config_name]["config"]) != config:
            raise ValueError("M03 configuration was not frozen on validation")
        verify_frozen_code(selection)
        if frozen.sha256_file(args.representation_manifest) != selection["representation_manifest_sha256"]:
            raise ValueError("M03 representation manifest changed after selection")
        if frozen.sha256_file(args.ai_representation_manifest) != selection["ai_representation_manifest_sha256"]:
            raise ValueError("M03 inherited AI manifest changed after selection")
    tafsir_manifest = load_representation_manifest(args.representation_manifest)
    ai_manifest = load_ai_manifest(args.ai_representation_manifest)
    fields = SCOPE_FIELDS[config["representation_scope"]]
    deployment, paths, snapshot = load_or_build_indexes(
        args, cases, fixture_manifest, tafsir_manifest, ai_manifest,
        load_ai="ai" in fields, load_tafsir="en_tafsir" in fields,
    )
    report = config_report(args, cases, fixture_manifest, deployment, paths, snapshot, tafsir_manifest, ai_manifest, config)
    if args.selection:
        for field in fields:
            if report["index_artifacts"][field] != selection["index_artifacts"][field]:
                raise ValueError(f"M03 {field} index artifacts changed after selection")
        report["selection_sha256"] = frozen.sha256_file(args.selection)
        report["selection"] = selection
        report["representation_manifest_sha256"] = frozen.sha256_file(args.representation_manifest)
        report["ai_representation_manifest_sha256"] = frozen.sha256_file(args.ai_representation_manifest)
    frozen.write_json(args.output, report)
    print(json.dumps({"output": str(args.output), "config": config, "queries": len(cases)}, indent=2))


def source_attribution_valid(report):
    try:
        allowed = set(SCOPE_FIELDS[report["config"]["representation_scope"]])
        runs = report["runs"]
        if not isinstance(runs, dict):
            return False
        for run in runs.values():
            if not isinstance(run, dict):
                return False
            refs = run.get("refs")
            source_fields = run.get("source_fields")
            if not isinstance(refs, list) or not isinstance(source_fields, list) or len(source_fields) != len(refs):
                return False
            if not all(
                isinstance(sources, list) and len(sources) == 1
                and isinstance(sources[0], str) and sources[0] in allowed
                for sources in source_fields
            ):
                return False
        return True
    except (AttributeError, KeyError, TypeError):
        return False


def _comparison(name, cases, baseline_scores, candidate_scores, baseline_rankings, candidate_rankings, bootstrap, seed):
    useful = [case for case in cases if any(grade >= 2 for grade in case["qrels"].values())]
    result = _paired_metrics(cases, baseline_scores, candidate_scores, bootstrap, seed)
    result.update({
        "positive_rank_regressions@10": frozen.rank_regression_details(cases, baseline_rankings, candidate_rankings, 1),
        "useful_rank_regressions@10": frozen.rank_regression_details(useful, baseline_rankings, candidate_rankings, 2),
    })
    return result


def _write_query_rankings(path: Path, cases, rankings, report):
    frozen.write_jsonl(path, (
        {
            "id": case["id"], "group_id": case["group_id"], "query": case["query"],
            "ranking": rankings[case["id"]], "source_fields": report["runs"][case["id"]]["source_fields"],
        }
        for case in cases
    ))


def build_report(args) -> None:
    cases, fixture_manifest = frozen.load_fixture(args.output_dir, "test")
    corpus = frozen.load_corpus(args.corpus_db)
    corpus_refs = set(corpus)
    score_group = frozen.scorer_module(args.scorer_package)
    loaded = {}
    for name, path in (
        ("candidate", args.candidate), ("off", args.off_candidate), ("tafsir", args.tafsir_candidate),
        ("ai", args.ai_candidate), ("joint", args.joint_candidate),
    ):
        report, rankings = frozen.load_run(path, cases, corpus_refs, fixture_manifest)
        loaded[name] = {"path": path, "report": report, "rankings": rankings, "scores": frozen.score_run(cases, rankings, score_group)}
    selection = json.loads(args.selection.read_text(encoding="utf8"))
    parent_report = json.loads((args.baseline_dir / "report.json").read_text(encoding="utf8"))
    if parent_report.get("experiment") != "M02" or parent_report.get("outcome") != "accepted" or parent_report.get("accepted_run") != "m01":
        raise ValueError("M03 parent is not the accepted M02 infrastructure stack")
    parent_artifact = parent_report["query_ranking_artifacts"]["m02"]
    parent_path = args.baseline_dir / parent_artifact["file"]
    if frozen.sha256_file(parent_path) != parent_artifact["sha256"]:
        raise ValueError("M02 committed ranking hash mismatch")
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
    scope = candidate_summary["provenance"]["config"]["representation_scope"]
    candidate_summary["leaderboard_label"] = f"M03 {scope}"
    useful = [case for case in cases if any(grade >= 2 for grade in case["qrels"].values())]
    comparisons = {}
    for candidate_name, baseline_name, baseline_scores, baseline_rankings in (
        ("candidate", "m02", parent_scores, parent_rankings),
        ("candidate", "browser", browser_scores, browser_rankings),
        ("off", "m02", parent_scores, parent_rankings),
        ("tafsir", "m02", parent_scores, parent_rankings),
        ("ai", "m02", parent_scores, parent_rankings),
        ("joint", "m02", parent_scores, parent_rankings),
        ("joint", "ai", loaded["ai"]["scores"], loaded["ai"]["rankings"]),
    ):
        comparisons[f"m03_{candidate_name}_vs_{baseline_name}"] = _comparison(
            candidate_name, cases, baseline_scores, loaded[candidate_name]["scores"],
            baseline_rankings, loaded[candidate_name]["rankings"], args.bootstrap, args.seed,
        )
    test_interaction = {
        metric: _interaction_details(
            cases, loaded["off"]["scores"], loaded["tafsir"]["scores"],
            loaded["ai"]["scores"], loaded["joint"]["scores"], metric, args.bootstrap, args.seed,
        )
        for metric in ("positive_hit@5", "useful_hit@5", "ndcg@10")
    }
    candidate_counts = frozen.headline_test_counts(candidate_summary)
    parent_counts = frozen.headline_test_counts(parent_report["runs"]["m02"])
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
        selection.get("experiment") == "M03" and selection.get("split") == "validation"
        and selection.get("test_accessed") is False
        and selection["dataset"]["fixture_sha256"] == frozen.EXPECTED_VALIDATION_FIXTURE_SHA256
        and selection["determinism_check"]["passed"]
    )
    try:
        verify_frozen_code(selection)
        code_unchanged = True
    except ValueError:
        code_unchanged = False
    parent_provenance = parent_report["runs"]["m02"]["provenance"]
    fixed_keys = ("model", "model_revision", "model_files", "dtype", "distance", "field", "display_field", "source_attribution", "corpus", "query_prefix", "passage_prefix")
    fixed_match = all(loaded["candidate"]["report"].get(key) == parent_provenance.get(key) for key in fixed_keys)
    manifest_match = loaded["candidate"]["report"].get("representation_manifest_sha256") == frozen.sha256_file(args.representation_manifest)
    ai_manifest_match = loaded["candidate"]["report"].get("ai_representation_manifest_sha256") == selection["ai_representation_manifest_sha256"]
    expected_configs = {
        "candidate": selected_config,
        "off": CONFIGS["parent_translation"],
        "tafsir": CONFIGS["parent_tafsir"],
        "ai": CONFIGS["parent_ai"],
        "joint": CONFIGS["parent_ai_tafsir"],
    }
    contract_flags = {}
    for name, value in loaded.items():
        report = value["report"]
        fields = SCOPE_FIELDS[canonical_config(report["config"])["representation_scope"]]
        contract_flags[name] = (
            canonical_config(report["config"]) == expected_configs[name]
            and report.get("selection_sha256") == selection_hash
            and report.get("selection") == selection
            and report.get("representation_manifest_sha256") == selection["representation_manifest_sha256"]
            and report.get("ai_representation_manifest_sha256") == selection["ai_representation_manifest_sha256"]
            and source_attribution_valid(report)
            and all(report.get("index_artifacts", {}).get(field) == selection["index_artifacts"].get(field) for field in fields)
        )
    off_exact = sum(loaded["off"]["rankings"][case["id"]] == parent_rankings[case["id"]] for case in cases)
    off_parity = off_exact == len(cases) and contract_flags["off"]
    validation_gate_passed = not selection["selected_changed"] or selection["validation_gate_passed"]
    checks = json.loads(args.checks.read_text(encoding="utf8"))
    accepted = all((
        candidate_config == selected_config, selection_contract, code_unchanged, fixed_match,
        manifest_match, ai_manifest_match, contract_flags["candidate"], contract_flags["tafsir"],
        contract_flags["ai"], contract_flags["joint"], test_gate, runtime_gate, off_parity,
        validation_gate_passed, checks.get("status") == "passed",
    ))
    if accepted and selection["selected_changed"]:
        candidate_summary["leaderboard_label"] = f"M03 {scope} (accepted)"
    elif not accepted:
        candidate_summary["leaderboard_label"] = f"M03 {scope} (rejected; parent retained)"
    else:
        candidate_summary["leaderboard_label"] = "M03 translation (parent retained)"
    report = {
        "profile": "pr2_frozen_v1",
        "experiment": "M03",
        "outcome": "accepted" if accepted else "rejected",
        "accepted_run": "m03" if accepted and selection["selected_changed"] else "m01",
        "parent_commit": "f222e26af8fb6fdc3251e27de763ebf2e1713cdd",
        "contribution": "Optional existing English Ibn Kathir tafsir with stable ayah IDs, empty-source accounting, repeated-block diagnostics and per-ayah max-score fusion. AI+tafsir is diagnostic and never promoted as the incremental M03 change.",
        "dataset": fixture_manifest,
        "selection": selection,
        "selection_sha256": selection_hash,
        "representation_manifest": selection["representation_manifest"],
        "representation_manifest_sha256": selection["representation_manifest_sha256"],
        "ai_representation_manifest_sha256": selection["ai_representation_manifest_sha256"],
        "parent_report_sha256": frozen.sha256_file(args.baseline_dir / "report.json"),
        "scorer_sha256": frozen.text_sha256(args.scorer_package / "evaluate_search.py"),
        "code_sha256": m03_code_hashes(),
        "runs": {**parent_report["runs"], "m03": candidate_summary},
        "diagnostic_runs": {"off": summaries["off"], "parent_tafsir": summaries["tafsir"], "parent_ai": summaries["ai"], "parent_ai_tafsir": summaries["joint"]},
        "comparisons": comparisons,
        "joint_interaction": test_interaction,
        "on_off_ablation": {
            "off": CONFIGS["parent_translation"],
            "on": selected_config,
            "off_exact_parent_queries": off_exact,
            "candidate_exact_off_queries": sum(loaded["candidate"]["rankings"][case["id"]] == loaded["off"]["rankings"][case["id"]] for case in cases),
            "candidate_exact_parent_queries": sum(loaded["candidate"]["rankings"][case["id"]] == parent_rankings[case["id"]] for case in cases),
            "off_headline_counts": frozen.headline_test_counts(summaries["off"]),
            "off_timings_ms": summaries["off"]["timings_ms"],
            "off_index_bytes": summaries["off"]["index_bytes"],
        },
        "runtime_budgets": parent_report["runtime_budgets"],
        "runtime_budget_check": {"budget": budget, "measurements": measurements, "passed": runtime_gate},
        "acceptance": {
            "accepted": accepted,
            "config_matches_frozen_selection": candidate_config == selected_config,
            "validation_only_selection_contract": selection_contract,
            "frozen_code_unchanged": code_unchanged,
            "fixed_model_corpus_contract_matches": fixed_match,
            "tafsir_manifest_matches": manifest_match,
            "ai_manifest_matches": ai_manifest_match,
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
            "English tafsir is explanatory content, not scripture; downstream reuse rights remain unresolved.",
            "Repeated commentary spans are audited but do not add document votes.",
            "Exact long-span screening does not certify semantic independence from source answers.",
            "The legacy train source has historical validation/test exposure; M03 export and selection did not use public test sources.",
            "The English IslamQA score does not establish Arabic/Bengali semantic quality.",
            "AI+tafsir results are interaction diagnostics, not an accepted M03 configuration.",
            "Runtime budgets remain fixed from M00.",
        ],
    }
    ranking_outputs = {
        "m03": (args.output_dir / "m03_query_rankings.jsonl", "candidate"),
        "m03_off": (args.output_dir / "m03_off_query_rankings.jsonl", "off"),
        "m03_tafsir": (args.output_dir / "m03_tafsir_query_rankings.jsonl", "tafsir"),
        "m03_ai": (args.output_dir / "m03_ai_query_rankings.jsonl", "ai"),
        "m03_joint": (args.output_dir / "m03_joint_query_rankings.jsonl", "joint"),
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
    export_parser = subparsers.add_parser("m03-export", help="Export and audit pinned English tafsir")
    export_parser.add_argument("--corpus-db", type=Path, default=frozen.DEFAULT_AAYATUN_REPO / "db" / "en_quran.db")
    export_parser.add_argument("--tafsir-db", type=Path, default=frozen.DEFAULT_AAYATUN_REPO / "db" / "en_kathir.db")
    export_parser.add_argument("--train-sources", type=Path, default=ROOT / "datasets" / "islamqa_references_train.json")
    export_parser.add_argument("--validation-questions", type=Path, default=frozen.DEFAULT_PACKAGE / "validation" / "questions.jsonl")
    export_parser.add_argument("--output-dir", type=Path, default=DEFAULT_M03_VALIDATION)
    export_parser.add_argument("--output", type=Path, default=DEFAULT_M03_VALIDATION / "runtime" / "en_tafsir_docs.jsonl")
    export_parser.set_defaults(func=export_tafsir)

    sweep_parser = subparsers.add_parser("m03-sweep", help="Select M03 on validation only")
    sweep_parser.add_argument("--corpus-db", type=Path, default=frozen.DEFAULT_AAYATUN_REPO / "db" / "en_quran.db")
    sweep_parser.add_argument("--scorer-package", type=Path, default=frozen.DEFAULT_SCORER_PACKAGE)
    sweep_parser.add_argument("--output-dir", type=Path, default=DEFAULT_M03_VALIDATION)
    sweep_parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_M03_VALIDATION / "runtime")
    sweep_parser.add_argument("--tafsir-output", type=Path, default=DEFAULT_M03_VALIDATION / "runtime" / "en_tafsir_docs.jsonl")
    sweep_parser.add_argument("--ai-output", type=Path, default=DEFAULT_M02_VALIDATION / "runtime" / "ai_docs.jsonl")
    sweep_parser.add_argument("--representation-manifest", type=Path, default=DEFAULT_M03_VALIDATION / "representation_manifest.json")
    sweep_parser.add_argument("--ai-representation-manifest", type=Path, default=DEFAULT_M02_VALIDATION / "representation_manifest.json")
    sweep_parser.add_argument("--selection", type=Path, default=DEFAULT_M03_VALIDATION / "selection.json")
    sweep_parser.add_argument("--bootstrap", type=int, default=2000)
    sweep_parser.add_argument("--seed", type=int, default=4213)
    sweep_parser.set_defaults(func=run_sweep)

    config_parser = subparsers.add_parser("m03-config", help="Run one frozen M03 configuration")
    config_parser.add_argument("--config-name", choices=tuple(CONFIGS), required=True)
    config_parser.add_argument("--corpus-db", type=Path, default=frozen.DEFAULT_AAYATUN_REPO / "db" / "en_quran.db")
    config_parser.add_argument("--output-dir", type=Path, default=DEFAULT_M03_TEST)
    config_parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_M03_VALIDATION / "runtime")
    config_parser.add_argument("--tafsir-output", type=Path, default=DEFAULT_M03_VALIDATION / "runtime" / "en_tafsir_docs.jsonl")
    config_parser.add_argument("--ai-output", type=Path, default=DEFAULT_M02_VALIDATION / "runtime" / "ai_docs.jsonl")
    config_parser.add_argument("--representation-manifest", type=Path, default=DEFAULT_M03_VALIDATION / "representation_manifest.json")
    config_parser.add_argument("--ai-representation-manifest", type=Path, default=DEFAULT_M02_VALIDATION / "representation_manifest.json")
    config_parser.add_argument("--output", type=Path, required=True)
    config_parser.add_argument("--split", choices=("test", "validation"), default="test")
    config_parser.add_argument("--selection", type=Path)
    config_parser.set_defaults(func=run_config)

    report_parser = subparsers.add_parser("m03-report", help="Score frozen M03 and append its leaderboard column")
    report_parser.add_argument("--corpus-db", type=Path, default=frozen.DEFAULT_AAYATUN_REPO / "db" / "en_quran.db")
    report_parser.add_argument("--scorer-package", type=Path, default=frozen.DEFAULT_SCORER_PACKAGE)
    report_parser.add_argument("--output-dir", type=Path, default=DEFAULT_M03_TEST)
    report_parser.add_argument("--candidate", type=Path, default=DEFAULT_M03_TEST / "m03_rankings.json")
    report_parser.add_argument("--off-candidate", type=Path, default=DEFAULT_M03_TEST / "m03_off_rankings.json")
    report_parser.add_argument("--tafsir-candidate", type=Path, default=DEFAULT_M03_TEST / "m03_tafsir_rankings.json")
    report_parser.add_argument("--ai-candidate", type=Path, default=DEFAULT_M03_TEST / "m03_ai_rankings.json")
    report_parser.add_argument("--joint-candidate", type=Path, default=DEFAULT_M03_TEST / "m03_joint_rankings.json")
    report_parser.add_argument("--selection", type=Path, default=DEFAULT_M03_VALIDATION / "selection.json")
    report_parser.add_argument("--representation-manifest", type=Path, default=DEFAULT_M03_VALIDATION / "representation_manifest.json")
    report_parser.add_argument("--ai-representation-manifest", type=Path, default=DEFAULT_M02_VALIDATION / "representation_manifest.json")
    report_parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_M02_TEST)
    report_parser.add_argument("--browser-baseline-dir", type=Path, default=frozen.DEFAULT_OUTPUT)
    report_parser.add_argument("--checks", type=Path, default=DEFAULT_M03_TEST / "checks.json")
    report_parser.add_argument("--bootstrap", type=int, default=2000)
    report_parser.add_argument("--seed", type=int, default=4213)
    report_parser.set_defaults(func=build_report)
