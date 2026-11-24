"""M02 validation, frozen regression, and reporting for optional Aayatun AI text."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

try:
    from . import islamqa_frozen as frozen
    from . import representations
except ImportError:
    import islamqa_frozen as frozen
    import representations


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_M02_VALIDATION = HERE / "results" / "m02_ai" / "validation"
DEFAULT_M02_TEST = HERE / "results" / "m02_ai" / "test"
DEFAULT_M01_TEST = HERE / "results" / "m01_retrieval" / "test"
CONFIGS = {
    "parent_translation": {"semantic_ai": False, "representation_scope": "translation"},
    "ai_only_diagnostic": {"semantic_ai": True, "representation_scope": "ai"},
    "parent_ai": {"semantic_ai": True, "representation_scope": "translation_ai"},
}


def canonical_config(config: dict[str, Any]) -> dict[str, Any]:
    semantic_ai = config["semantic_ai"]
    if not isinstance(semantic_ai, bool):
        raise ValueError("semantic_ai must be a boolean")
    scope = config["representation_scope"]
    if scope not in ("translation", "ai", "translation_ai"):
        raise ValueError("Unknown M02 representation scope")
    if scope != "translation" and not semantic_ai:
        raise ValueError("AI scopes require semantic_ai=true")
    return {"semantic_ai": semantic_ai, "representation_scope": scope}


def config_sha256(config: dict[str, Any]) -> str:
    payload = json.dumps(canonical_config(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf8")).hexdigest()


def export_ai(args) -> None:
    from transformers import AutoTokenizer

    corpus = frozen.load_corpus(args.corpus_db)
    if len(corpus) != 6236 or frozen.corpus_hash(corpus) != frozen.EXPECTED_CORPUS_SHA256:
        raise ValueError("Corpus does not match the frozen fixture")
    tokenizer = AutoTokenizer.from_pretrained(frozen.model_snapshot(), local_files_only=True)
    manifest = representations.export_ai_representation(
        args.ai_db,
        corpus,
        args.output,
        [args.train_sources, args.benchmark_sources],
        tokenizer=tokenizer,
        max_tokens=512,
    )
    truncation = manifest.get("token_truncation") or {}
    if truncation.get("truncated_ayahs_file"):
        sidecar = args.output.parent / truncation["truncated_ayahs_file"]
        truncation["truncated_ayahs_file"] = Path(os.path.relpath(sidecar, args.output_dir)).as_posix()
    frozen.write_json(args.output_dir / "representation_manifest.json", manifest)
    print(json.dumps({
        "output": str(args.output),
        "manifest": str(args.output_dir / "representation_manifest.json"),
        "counts": manifest["counts"],
        "truncated": manifest["token_truncation"]["truncated_documents"],
        "overlaps": manifest["provenance_audit"]["overlap_count"],
    }, indent=2))


def load_representation_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf8"))
    if manifest.get("experiment") != "M02" or manifest.get("representation") != "ai":
        raise ValueError("Invalid M02 representation manifest")
    if manifest["source"]["source_db_sha256"] != representations.AI_DB_SHA256:
        raise ValueError("M02 AI source hash changed")
    if manifest["corpus"]["sha256"] != frozen.EXPECTED_CORPUS_SHA256:
        raise ValueError("M02 representation corpus hash changed")
    truncation = manifest.get("token_truncation") or {}
    if truncation.get("truncated_documents"):
        sidecar = path.parent / truncation["truncated_ayahs_file"]
        if frozen.sha256_file(sidecar) != truncation["truncated_ayahs_sha256"]:
            raise ValueError("M02 truncated-ayah sidecar hash mismatch")
    return manifest


def load_or_build_ai_index(args, cases, fixture_manifest, manifest, load_ai=True):
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document

    deployment, translation_path, snapshot = frozen.load_or_build_dense_index(args, cases, fixture_manifest)
    rows = frozen.read_jsonl(args.ai_output)
    if frozen.sha256_file(args.ai_output) != manifest["output"]["sha256"]:
        raise ValueError("M02 AI export hash mismatch")
    refs = [row["ayah"] for row in rows]
    if len(refs) != len(set(refs)) or set(refs) - set(frozen.load_corpus(args.corpus_db)):
        raise ValueError("M02 AI export has duplicate or unknown ayat")
    if any(row.get("field") != "ai" or row.get("source") != "aayatun_ai_generated" for row in rows):
        raise ValueError("M02 AI source attribution mismatch")
    ai_path = args.runtime_dir / "e5-small" / "quran-ai"
    if not load_ai:
        return deployment, translation_path, ai_path, snapshot

    ai_path.parent.mkdir(parents=True, exist_ok=True)
    expected = {f"ai:{row['ayah']}": row for row in rows}
    if not (ai_path / "index.faiss").exists():
        documents = [
            Document(
                page_content="passage: " + row["text"],
                metadata={
                    "surah": int(row["ayah"].split(":")[0]),
                    "ayah": int(row["ayah"].split(":")[1]),
                    "field": "ai",
                    "source": "aayatun_ai_generated",
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
        database.save_local(str(ai_path))
    deployment.ai_db = FAISS.load_local(
        str(ai_path), deployment.embeddings, normalize_L2=True, allow_dangerous_deserialization=True
    )
    deployment.ai_db_docstore_id_to_index = dict(
        zip(deployment.ai_db.index_to_docstore_id.values(), deployment.ai_db.index_to_docstore_id.keys())
    )
    if deployment.ai_db.index.ntotal != len(rows):
        raise ValueError("M02 AI index size mismatch")
    if set(deployment.ai_db.index_to_docstore_id.values()) != set(expected):
        raise ValueError("M02 AI index document IDs mismatch")
    for doc_id, row in expected.items():
        document = deployment.ai_db.docstore.search(doc_id)
        surah, ayah = (int(part) for part in row["ayah"].split(":"))
        expected_metadata = {
            "surah": surah,
            "ayah": ayah,
            "field": "ai",
            "source": "aayatun_ai_generated",
            "chunk": 0,
        }
        if (
            document is None
            or document.page_content != "passage: " + row["text"]
            or document.metadata != expected_metadata
        ):
            raise ValueError(f"M02 AI index content mismatch: {row['ayah']}")
    return deployment, translation_path, ai_path, snapshot


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
            representation_scope=config["representation_scope"],
        )
        elapsed = round((time.perf_counter() - started) * 1000, 3)
        refs = [f"{doc.metadata['surah']}:{doc.metadata['ayah']}" for doc, _ in results]
        fields = [doc.metadata["field"] for doc, _ in results]
        if len(refs) != len(set(refs)) or len(refs) > 50:
            raise ValueError(f"M02 duplicate or over-depth result: {case['id']}")
        expected = {
            "translation": {"translation"},
            "ai": {"ai"},
            "translation_ai": {"translation", "ai"},
        }[config["representation_scope"]]
        if set(fields) - expected:
            raise ValueError(f"M02 source-field leakage: {case['id']}")
        runs[case["id"]] = {"refs": refs, "source_fields": [[field] for field in fields], "ms": elapsed}
        if index % 50 == 0:
            print(f"M02 {config['representation_scope']}: {index}/{len(cases)}", flush=True)
    return {"config": config, "result_k": 50, "runs": runs}


def config_report(args, cases, fixture_manifest, deployment, translation_path, ai_path, snapshot, config):
    execution = execute_config(deployment, cases, config)
    scope = execution["config"]["representation_scope"]
    active_paths = []
    if scope in ("translation", "translation_ai"):
        active_paths.append(translation_path)
    if scope in ("ai", "translation_ai"):
        active_paths.append(ai_path)
    report = {
        "engine": "M02 offline multilingual E5-small float32 FAISS with optional AI representation",
        "profile": "pr2_frozen_v1",
        "experiment": "M02",
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
        "index_bytes": sum(index_bytes(path) for path in active_paths),
        "index_artifacts": {
            "translation": index_artifacts(translation_path),
            "ai": index_artifacts(ai_path),
        },
        "active_representations": [path.name for path in active_paths],
        "peak_rss_bytes": frozen.peak_rss_bytes(),
        **execution,
    }
    report["config_sha256"] = config_sha256(report["config"])
    return report


def validation_summary(name, report, cases, corpus_refs, fixture_manifest, score_group, bootstrap, seed):
    rankings = frozen.validate_run(cases, report, corpus_refs)
    scores = frozen.score_run(cases, rankings, score_group)
    useful = [case for case in cases if any(grade >= 2 for grade in case["qrels"].values())]
    direct = [case for case in cases if case["best_settled_label"] == "direct"]
    headline = {}
    for slice_name, selected in (("all", cases), ("useful", useful), ("direct", direct)):
        row = frozen.aggregate(selected, scores, bootstrap, seed)
        row["counts"] = {
            metric: sum(scores[case["id"]][metric] > 0 for case in selected)
            for metric in ("positive_hit@5", "useful_hit@5")
        }
        headline[slice_name] = row
    summary = {
        "name": name,
        "config": report["config"],
        "config_sha256": report["config_sha256"],
        "headline": headline,
        "timings_ms": frozen.timing_summary(report, cases),
        "coverage@10": frozen.result_coverage(cases, rankings, 10),
        "index_bytes": report["index_bytes"],
        "peak_rss_bytes": report["peak_rss_bytes"],
    }
    return summary, rankings, scores


def m02_code_hashes():
    return {
        "evaluation/islamqa_frozen.py": frozen.sha256_file(HERE / "islamqa_frozen.py"),
        "evaluation/m02_ai.py": frozen.sha256_file(HERE / "m02_ai.py"),
        "evaluation/representations.py": frozen.sha256_file(HERE / "representations.py"),
        "evaluation/test_ai_representation.py": frozen.sha256_file(HERE / "test_ai_representation.py"),
        "evaluation/test_retrieval_config.py": frozen.sha256_file(HERE / "test_retrieval_config.py"),
        "evaluation/test_islamqa_frozen.py": frozen.sha256_file(HERE / "test_islamqa_frozen.py"),
        "utils.py": frozen.sha256_file(ROOT / "utils.py"),
        "build_vector_database.py": frozen.sha256_file(ROOT / "build_vector_database.py"),
    }


def verify_frozen_code(selection):
    for relative, digest in selection["code_sha256"].items():
        if frozen.sha256_file(ROOT / relative) != digest:
            raise ValueError(f"Frozen M02 code changed after selection: {relative}")


def run_sweep(args) -> None:
    cases, fixture_manifest = frozen.load_fixture(args.output_dir, "validation")
    corpus = frozen.load_corpus(args.corpus_db)
    score_group = frozen.scorer_module(args.scorer_package)
    manifest = load_representation_manifest(args.representation_manifest)
    deployment, translation_path, ai_path, snapshot = load_or_build_ai_index(
        args, cases, fixture_manifest, manifest
    )
    summaries = {}
    scores_by_name = {}
    artifacts = {}
    for name, config in CONFIGS.items():
        report = config_report(args, cases, fixture_manifest, deployment, translation_path, ai_path, snapshot, config)
        path = args.output_dir / "validation_runs" / f"{name}.json"
        frozen.write_json(path, report)
        summary, _, scores = validation_summary(
            name, report, cases, set(corpus), fixture_manifest, score_group, args.bootstrap, args.seed
        )
        summaries[name] = summary
        scores_by_name[name] = scores
        artifacts[name] = {"file": str(path.relative_to(args.output_dir)), "sha256": frozen.sha256_file(path)}
    parent = summaries["parent_translation"]
    candidate = summaries["parent_ai"]
    selected = candidate if frozen.validation_gate(parent, candidate) else parent
    useful = [case for case in cases if any(grade >= 2 for grade in case["qrels"].values())]
    marginal_effects = {
        name: {
            metric: frozen.paired_details(
                useful if metric == "useful_hit@5" else cases,
                scores_by_name[parent["name"]], scores_by_name[name], metric, args.bootstrap, args.seed,
            )
            for metric in ("positive_hit@5", "useful_hit@5", "ndcg@10")
        }
        for name in CONFIGS
    }
    repeat = execute_config(deployment, cases, selected["config"])
    selected_path = args.output_dir / "validation_runs" / f"{selected['name']}.json"
    selected_report = json.loads(selected_path.read_text(encoding="utf8"))
    exact = sum(
        repeat["runs"][case["id"]]["refs"] == selected_report["runs"][case["id"]]["refs"]
        and repeat["runs"][case["id"]]["source_fields"] == selected_report["runs"][case["id"]]["source_fields"]
        for case in cases
    )
    if exact != len(cases):
        raise ValueError("M02 selected configuration is not deterministic")
    selection = {
        "profile": "pr2_frozen_v1",
        "experiment": "M02",
        "split": "validation",
        "dataset": fixture_manifest,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_accessed": False,
        "selection_policy": "Promote parent+AI only if validation useful hit@5 increases and validation nDCG@10 does not decrease. AI-only is diagnostic and is not promotion-eligible because translation must be retained.",
        "candidates": summaries,
        "marginal_effects": marginal_effects,
        "selected_name": selected["name"],
        "selected_config": selected["config"],
        "selected_config_sha256": selected["config_sha256"],
        "selected_changed": selected["config_sha256"] != parent["config_sha256"],
        "validation_gate_passed": frozen.validation_gate(parent, candidate),
        "determinism_check": {"queries": len(cases), "exact_queries": exact, "passed": exact == len(cases)},
        "representation_manifest": manifest,
        "representation_manifest_sha256": frozen.sha256_file(args.representation_manifest),
        "ai_index_artifacts": index_artifacts(ai_path),
        "model": {"name": frozen.E5_MODEL, "revision": frozen.E5_REVISION, "files": frozen.dense_model_files(snapshot)},
        "prefixes": {"query": "query: ", "passage": "passage: "},
        "code_sha256": m02_code_hashes(),
        "validation_ranking_artifacts": artifacts,
    }
    frozen.write_json(args.selection, selection)
    print(json.dumps({"selected": selection["selected_config"], "gate": selection["validation_gate_passed"], "candidates": list(summaries)}, indent=2))


def run_config(args) -> None:
    config = canonical_config({"semantic_ai": args.semantic_ai == "true", "representation_scope": args.representation_scope})
    cases, fixture_manifest = frozen.load_fixture(args.output_dir, args.split)
    if args.selection:
        selection = json.loads(args.selection.read_text(encoding="utf8"))
        if selection.get("split") != "validation" or canonical_config(selection["selected_config"]) != config:
            raise ValueError("M02 test configuration does not match frozen validation selection")
        verify_frozen_code(selection)
        if frozen.sha256_file(args.representation_manifest) != selection["representation_manifest_sha256"]:
            raise ValueError("M02 representation manifest changed after selection")
    manifest = load_representation_manifest(args.representation_manifest)
    deployment, translation_path, ai_path, snapshot = load_or_build_ai_index(
        args, cases, fixture_manifest, manifest, load_ai=config["representation_scope"] != "translation"
    )
    report = config_report(args, cases, fixture_manifest, deployment, translation_path, ai_path, snapshot, config)
    if args.selection:
        if config["representation_scope"] != "translation" and report["index_artifacts"]["ai"] != selection["ai_index_artifacts"]:
            raise ValueError("M02 AI index artifacts changed after selection")
        report["selection_sha256"] = frozen.sha256_file(args.selection)
        report["selection"] = selection
        report["representation_manifest_sha256"] = frozen.sha256_file(args.representation_manifest)
    frozen.write_json(args.output, report)
    print(json.dumps({"output": str(args.output), "config": config, "queries": len(cases)}, indent=2))


def headline_counts(summary):
    return {
        "relevant_hit@5": summary["headline"]["all"]["counts"]["positive_hit@5"],
        "useful_hit@5": summary["headline"]["useful"]["counts"]["useful_hit@5"],
        "direct_useful_hit@5": summary["headline"]["direct"]["counts"]["useful_hit@5"],
        "ndcg@10": summary["headline"]["all"]["metrics"]["ndcg@10"]["mean"],
    }


def source_attribution_valid(report):
    scope = report["config"]["representation_scope"]
    allowed = {
        "translation": {"translation"},
        "ai": {"ai"},
        "translation_ai": {"translation", "ai"},
    }[scope]
    for run in report["runs"].values():
        refs = run.get("refs")
        source_fields = run.get("source_fields")
        if not isinstance(refs, list) or not isinstance(source_fields, list) or len(source_fields) != len(refs):
            return False
        if not all(
            isinstance(sources, list)
            and len(sources) == 1
            and isinstance(sources[0], str)
            and sources[0] in allowed
            for sources in source_fields
        ):
            return False
    return True


def build_report(args) -> None:
    cases, fixture_manifest = frozen.load_fixture(args.output_dir, "test")
    corpus = frozen.load_corpus(args.corpus_db)
    corpus_refs = set(corpus)
    score_group = frozen.scorer_module(args.scorer_package)
    candidate_report, candidate_rankings = frozen.load_run(args.candidate, cases, corpus_refs, fixture_manifest)
    off_report, off_rankings = frozen.load_run(args.off_candidate, cases, corpus_refs, fixture_manifest)
    candidate_scores = frozen.score_run(cases, candidate_rankings, score_group)
    off_scores = frozen.score_run(cases, off_rankings, score_group)
    selection = json.loads(args.selection.read_text(encoding="utf8"))
    parent_report = json.loads((args.baseline_dir / "report.json").read_text(encoding="utf8"))
    if parent_report.get("experiment") != "M01" or parent_report.get("outcome") != "accepted":
        raise ValueError("M02 parent is not the accepted M01 report")
    parent_artifact = parent_report["query_ranking_artifacts"]["m01"]
    parent_path = args.baseline_dir / parent_artifact["file"]
    if frozen.sha256_file(parent_path) != parent_artifact["sha256"]:
        raise ValueError("M01 committed ranking hash mismatch")
    parent_rows = frozen.read_jsonl(parent_path)
    parent_rankings = {row["id"]: row["ranking"] for row in parent_rows}
    parent_scores = frozen.score_run(cases, parent_rankings, score_group)
    browser_report = json.loads((args.browser_baseline_dir / "report.json").read_text(encoding="utf8"))
    browser_rankings, _ = frozen.load_committed_rankings("browser", args.browser_baseline_dir, browser_report, cases, corpus_refs)
    browser_scores = frozen.score_run(cases, browser_rankings, score_group)
    candidate_summary = frozen.candidate_headline_summary(candidate_report, cases, candidate_rankings, candidate_scores, args.bootstrap, args.seed)
    candidate_summary["leaderboard_label"] = f"M02 {candidate_report['config']['representation_scope']}"
    off_summary = frozen.candidate_headline_summary(off_report, cases, off_rankings, off_scores, args.bootstrap, args.seed)
    summary_runs = {**parent_report["runs"], "m02": candidate_summary}
    useful = [case for case in cases if any(grade >= 2 for grade in case["qrels"].values())]
    comparisons = {}
    for name, baseline_scores, baseline_rankings in (
        ("m01", parent_scores, parent_rankings),
        ("browser", browser_scores, browser_rankings),
    ):
        comparisons[f"m02_vs_{name}"] = {
            metric: frozen.paired_details(
                useful if metric == "useful_hit@5" else cases,
                baseline_scores, candidate_scores, metric, args.bootstrap, args.seed,
            )
            for metric in ("positive_hit@5", "useful_hit@5", "ndcg@10")
        }
        comparisons[f"m02_vs_{name}"].update({
            "positive_rank_regressions@10": frozen.rank_regression_details(cases, baseline_rankings, candidate_rankings, 1),
            "useful_rank_regressions@10": frozen.rank_regression_details(useful, baseline_rankings, candidate_rankings, 2),
        })
    candidate_counts = headline_counts(candidate_summary)
    parent_counts = headline_counts(parent_report["runs"]["m01"])
    test_gate = all(candidate_counts[key] >= parent_counts[key] for key in parent_counts)
    budget = parent_report["runtime_budgets"]["runs"]["dense"]
    measurements = {
        "p95_query_ms": candidate_summary["timings_ms"]["p95"],
        "index_bytes": candidate_summary["index_bytes"],
        "peak_rss_bytes": candidate_summary["peak_rss_bytes"],
    }
    runtime_gate = all(measurements[key] <= budget[f"{key}_max"] for key in measurements)
    selected_config = canonical_config(selection["selected_config"])
    config_match = canonical_config(candidate_report["config"]) == selected_config
    selection_hash_match = candidate_report.get("selection_sha256") == frozen.sha256_file(args.selection)
    selection_match = candidate_report.get("selection") == selection
    manifest_match = candidate_report.get("representation_manifest_sha256") == frozen.sha256_file(args.representation_manifest)
    index_match = (
        candidate_report["config"]["representation_scope"] == "translation"
        or candidate_report.get("index_artifacts", {}).get("ai") == selection.get("ai_index_artifacts")
    )
    try:
        verify_frozen_code(selection)
        code_unchanged = True
    except ValueError:
        code_unchanged = False
    selection_contract = (
        selection.get("split") == "validation"
        and selection.get("test_accessed") is False
        and selection["dataset"]["fixture_sha256"] == frozen.EXPECTED_VALIDATION_FIXTURE_SHA256
        and selection["determinism_check"]["passed"]
    )
    parent_provenance = parent_report["runs"]["m01"]["provenance"]
    fixed_keys = ("model", "model_revision", "model_files", "dtype", "distance", "field", "source_attribution", "corpus")
    fixed_match = all(candidate_report.get(key) == parent_provenance.get(key) for key in fixed_keys if key != "source_attribution")
    prefix_match = candidate_report.get("query_prefix") == "query: " and candidate_report.get("passage_prefix") == "passage: "
    attribution_valid = source_attribution_valid(candidate_report)
    off_config = canonical_config(off_report["config"])
    off_exact = sum(off_rankings[case["id"]] == parent_rankings[case["id"]] for case in cases)
    off_parity = (
        off_exact == len(cases)
        and off_config == CONFIGS["parent_translation"]
        and source_attribution_valid(off_report)
    )
    validation_gate_passed = not selection["selected_changed"] or selection["validation_gate_passed"]
    checks = json.loads(args.checks.read_text(encoding="utf8"))
    accepted = all((
        config_match, selection_hash_match, selection_match, manifest_match, index_match, code_unchanged,
        selection_contract, fixed_match, prefix_match, attribution_valid, test_gate, runtime_gate,
        off_parity, validation_gate_passed, checks.get("status") == "passed",
    ))
    report = {
        "profile": "pr2_frozen_v1",
        "experiment": "M02",
        "outcome": "accepted" if accepted else "rejected",
        "accepted_run": "m02" if accepted and selection["selected_changed"] else "m01",
        "parent_commit": "0b24d40590394e49068b0aca38334de57508de9c",
        "contribution": "Optional existing Aayatun AI representation with per-ayah max-score fusion; AI-only is diagnostic. No relevance gain is claimed unless the frozen parent+AI candidate passes every gate.",
        "dataset": fixture_manifest,
        "selection": selection,
        "selection_sha256": frozen.sha256_file(args.selection),
        "representation_manifest": selection["representation_manifest"],
        "representation_manifest_sha256": selection["representation_manifest_sha256"],
        "parent_report_sha256": frozen.sha256_file(args.baseline_dir / "report.json"),
        "scorer_sha256": frozen.text_sha256(args.scorer_package / "evaluate_search.py"),
        "code_sha256": m02_code_hashes(),
        "runs": summary_runs,
        "comparisons": comparisons,
        "on_off_ablation": {
            "off": CONFIGS["parent_translation"],
            "on": selected_config,
            "off_exact_m01_queries": off_exact,
            "off_headline_counts": headline_counts(off_summary),
            "off_timings_ms": off_summary["timings_ms"],
            "off_index_bytes": off_summary["index_bytes"],
            "candidate_exact_off_queries": sum(candidate_rankings[case["id"]] == off_rankings[case["id"]] for case in cases),
        },
        "runtime_budgets": parent_report["runtime_budgets"],
        "runtime_budget_check": {"budget": budget, "measurements": measurements, "passed": runtime_gate},
        "acceptance": {
            "accepted": accepted,
            "config_matches_frozen_selection": config_match,
            "selection_hash_matches": selection_hash_match,
            "embedded_selection_matches": selection_match,
            "representation_manifest_matches": manifest_match,
            "ai_index_artifacts_match": index_match,
            "frozen_code_unchanged": code_unchanged,
            "validation_only_selection_contract": selection_contract,
            "fixed_model_corpus_contract_matches": fixed_match,
            "prefixes_match": prefix_match,
            "source_attribution_valid": attribution_valid,
            "validation_gate_passed": validation_gate_passed,
            "test_gate_passed": test_gate,
            "runtime_budget_passed": runtime_gate,
            "similarity_rollback_parity": off_parity,
            "targeted_checks_passed": checks.get("status") == "passed",
            "parent_headline_counts": parent_counts,
            "candidate_headline_counts": candidate_counts,
        },
        "targeted_checks": checks,
        "source_ranking_artifacts": {
            "m02": {
                "file": args.candidate.name,
                "sha256": frozen.sha256_file(args.candidate),
                "bytes": args.candidate.stat().st_size,
            },
            "m02_off": {
                "file": args.off_candidate.name,
                "sha256": frozen.sha256_file(args.off_candidate),
                "bytes": args.off_candidate.stat().st_size,
            },
        },
        "limitations": [
            *fixture_manifest["notes"],
            "AI explanations are generated content, not scripture; reuse rights remain unresolved.",
            "Exact long-span screening does not certify semantic independence from IslamQA answers.",
            "The English IslamQA score does not establish Arabic/Bengali semantic quality.",
            "Runtime budgets remain fixed from M00.",
        ],
    }
    frozen.write_jsonl(
        args.output_dir / "m02_query_rankings.jsonl",
        (
            {
                "id": case["id"], "group_id": case["group_id"], "query": case["query"],
                "ranking": candidate_rankings[case["id"]],
                "source_fields": candidate_report["runs"][case["id"]]["source_fields"],
            }
            for case in cases
        ),
    )
    report["query_ranking_artifacts"] = {
        "m02": {
            "file": "m02_query_rankings.jsonl",
            "sha256": frozen.sha256_file(args.output_dir / "m02_query_rankings.jsonl"),
            "bytes": (args.output_dir / "m02_query_rankings.jsonl").stat().st_size,
        }
    }
    table = frozen.markdown_leaderboard(report)
    (args.output_dir / "leaderboard.md").write_text(table + "\n", encoding="utf8")
    frozen.write_json(args.output_dir / "report.json", report)
    print(table)
    print(json.dumps(report["acceptance"], indent=2))


def add_parsers(subparsers) -> None:
    export_parser = subparsers.add_parser("m02-export", help="Export and audit pinned Aayatun AI explanations")
    export_parser.add_argument("--corpus-db", type=Path, default=frozen.DEFAULT_AAYATUN_REPO / "db" / "en_quran.db")
    export_parser.add_argument("--ai-db", type=Path, default=frozen.DEFAULT_PR2_EVAL_REPO / "db" / "ai_context.db")
    export_parser.add_argument("--train-sources", type=Path, default=ROOT / "datasets" / "islamqa_references_train.json")
    export_parser.add_argument("--benchmark-sources", type=Path, default=ROOT / "datasets" / "islamqa_references_benchmark.json")
    export_parser.add_argument("--output-dir", type=Path, default=DEFAULT_M02_VALIDATION)
    export_parser.add_argument("--output", type=Path, default=DEFAULT_M02_VALIDATION / "runtime" / "ai_docs.jsonl")
    export_parser.set_defaults(func=export_ai)

    sweep_parser = subparsers.add_parser("m02-sweep", help="Select M02 on validation only")
    sweep_parser.add_argument("--corpus-db", type=Path, default=frozen.DEFAULT_AAYATUN_REPO / "db" / "en_quran.db")
    sweep_parser.add_argument("--scorer-package", type=Path, default=frozen.DEFAULT_SCORER_PACKAGE)
    sweep_parser.add_argument("--output-dir", type=Path, default=DEFAULT_M02_VALIDATION)
    sweep_parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_M02_VALIDATION / "runtime")
    sweep_parser.add_argument("--ai-output", type=Path, default=DEFAULT_M02_VALIDATION / "runtime" / "ai_docs.jsonl")
    sweep_parser.add_argument("--representation-manifest", type=Path, default=DEFAULT_M02_VALIDATION / "representation_manifest.json")
    sweep_parser.add_argument("--selection", type=Path, default=DEFAULT_M02_VALIDATION / "selection.json")
    sweep_parser.add_argument("--bootstrap", type=int, default=2000)
    sweep_parser.add_argument("--seed", type=int, default=4213)
    sweep_parser.set_defaults(func=run_sweep)

    config_parser = subparsers.add_parser("m02-config", help="Run one frozen M02 configuration")
    config_parser.add_argument("--corpus-db", type=Path, default=frozen.DEFAULT_AAYATUN_REPO / "db" / "en_quran.db")
    config_parser.add_argument("--output-dir", type=Path, default=DEFAULT_M02_TEST)
    config_parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_M02_TEST / "runtime")
    config_parser.add_argument("--ai-output", type=Path, default=DEFAULT_M02_VALIDATION / "runtime" / "ai_docs.jsonl")
    config_parser.add_argument("--representation-manifest", type=Path, default=DEFAULT_M02_VALIDATION / "representation_manifest.json")
    config_parser.add_argument("--output", type=Path, default=DEFAULT_M02_TEST / "m02_rankings.json")
    config_parser.add_argument("--split", choices=("test", "validation"), default="test")
    config_parser.add_argument("--semantic-ai", choices=("false", "true"), required=True)
    config_parser.add_argument("--representation-scope", choices=("translation", "ai", "translation_ai"), required=True)
    config_parser.add_argument("--selection", type=Path)
    config_parser.set_defaults(func=run_config)

    report_parser = subparsers.add_parser("m02-report", help="Score frozen M02 and append its leaderboard column")
    report_parser.add_argument("--corpus-db", type=Path, default=frozen.DEFAULT_AAYATUN_REPO / "db" / "en_quran.db")
    report_parser.add_argument("--scorer-package", type=Path, default=frozen.DEFAULT_SCORER_PACKAGE)
    report_parser.add_argument("--output-dir", type=Path, default=DEFAULT_M02_TEST)
    report_parser.add_argument("--candidate", type=Path, default=DEFAULT_M02_TEST / "m02_rankings.json")
    report_parser.add_argument("--off-candidate", type=Path, default=DEFAULT_M02_TEST / "m02_off_rankings.json")
    report_parser.add_argument("--selection", type=Path, default=DEFAULT_M02_VALIDATION / "selection.json")
    report_parser.add_argument("--representation-manifest", type=Path, default=DEFAULT_M02_VALIDATION / "representation_manifest.json")
    report_parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_M01_TEST)
    report_parser.add_argument("--browser-baseline-dir", type=Path, default=frozen.DEFAULT_OUTPUT)
    report_parser.add_argument("--checks", type=Path, default=DEFAULT_M02_TEST / "checks.json")
    report_parser.add_argument("--bootstrap", type=int, default=2000)
    report_parser.add_argument("--seed", type=int, default=4213)
    report_parser.set_defaults(func=build_report)
