"""Frozen IslamQA v3 evaluation contract for the Aayatun PR #2 leaderboard."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import closing
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import random
import re
import sqlite3
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time
import types
from typing import Any, Callable, Iterable


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WORKSPACE = ROOT.parents[1]
DEFAULT_PACKAGE = ROOT / "datasets" / "sanitized_islamqa" / "split_package_v3"
DEFAULT_SCORER_PACKAGE = ROOT / "datasets" / "sanitized_islamqa" / "split_package"
DEFAULT_AAYATUN_REPO = WORKSPACE / "aayatun-research"
DEFAULT_PR2_EVAL_REPO = WORKSPACE / "aayatun-pr2-eval"
DEFAULT_OUTPUT = HERE / "results" / "m00_frozen_v1"
DEFAULT_M01_VALIDATION = HERE / "results" / "m01_retrieval" / "validation"
DEFAULT_M01_TEST = HERE / "results" / "m01_retrieval" / "test"

DATASET_COMMIT = "d936c64c51ebd0908b6ecf0d9b6e4debbb39b517"
DEPLOY_REVISION = "d9cff5d3f21dc11423465893fb1839dd6c9251c0"
FCS_REVISION = "54990b849f176d722fb087372c8875cb2d7c6e14"
PR2_REVISION = "357a367ef4631cb771cdd9e3a34116280c7dd2d8"
E5_MODEL = "intfloat/multilingual-e5-small"
E5_REVISION = "c007d7ef6fd86656326059b28395a7a03a7c5846"
FIXTURE_NAME = "islamqa_v3_test_queries.jsonl"
MANIFEST_NAME = "islamqa_v3_fixture_manifest.json"
EXPECTED_FIXTURE_SHA256 = "cb33c2b58d229caac4c8374af7a857cbd93a4578833592fcac1afd4e4ab5ab25"
EXPECTED_VALIDATION_FIXTURE_SHA256 = "1ae65f7cca2e2aa811b0b62e331d43fc98c78f62a7d23546ffa5109eab0a6cfe"
EXPECTED_CORPUS_SHA256 = "d1c40be0bef518288500f3940d4419b3516c2497b7285f5dcbd22802487e34e0"
AYAH_RE = re.compile(r"[1-9]\d*:[1-9]\d*\Z")

RUN_COLUMNS = (
    ("deploy", "Deploy keyword"),
    ("fcs_translation", "FCS Translation"),
    ("fcs_all", "FCS All-fields"),
    ("browser", "PR #2 browser"),
    ("dense", "M00 offline E5/FAISS"),
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    )
    path.write_bytes(payload.encode("utf8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf8")
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def corpus_hash(corpus: dict[str, str]) -> str:
    payload = json.dumps(corpus, ensure_ascii=False, sort_keys=True).encode("utf8")
    return hashlib.sha256(payload).hexdigest()


def load_corpus(path: Path) -> dict[str, str]:
    with closing(sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)) as db:
        rows = db.execute(
            "SELECT sura, ayah, text FROM verses ORDER BY sura, ayah"
        ).fetchall()
    corpus = {f"{surah}:{ayah}": text for surah, ayah, text in rows}
    if len(corpus) != len(rows):
        raise ValueError("Corpus contains duplicate ayah references")
    return corpus


def verify_v3_package(package: Path) -> dict[str, Any]:
    package = package.resolve()
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf8"))
    if manifest.get("dataset_version") != "3.0.0":
        raise ValueError("Expected IslamQA dataset v3.0.0")
    if manifest.get("schema_version") != 2:
        raise ValueError("Expected the unchanged v2 record schema")
    for relative, expected in manifest["outputs"].items():
        path = package / relative
        if sha256_file(path) != expected["sha256"]:
            raise ValueError(f"v3 hash mismatch: {relative}")
        if len(read_jsonl(path)) != expected["records"]:
            raise ValueError(f"v3 record count mismatch: {relative}")
    v2 = package.parent / "split_package"
    for split in ("validation", "test"):
        for name in ("questions.jsonl", "grades.jsonl", "unjudged_candidates.jsonl"):
            if (package / split / name).read_bytes() != (v2 / split / name).read_bytes():
                raise ValueError(f"v3 changed preserved {split}/{name}")
    expected_groups = {"train": 1086, "validation": 257, "test": 341}
    for split, expected in expected_groups.items():
        actual = len(read_jsonl(package / split / "questions.jsonl"))
        if actual != expected:
            raise ValueError(f"{split}: expected {expected} groups, got {actual}")
    return manifest


def load_cases(package: Path, split: str = "test") -> list[dict[str, Any]]:
    questions = read_jsonl(package / split / "questions.jsonl")
    grades = read_jsonl(package / split / "grades.jsonl")
    by_group: dict[str, dict[str, int]] = defaultdict(dict)
    question_ids = {row["group_id"] for row in questions}
    for row in grades:
        group_id = row["group_id"]
        if group_id not in question_ids:
            raise ValueError(f"Qrel references unknown group: {group_id}")
        if row["ayah"] in by_group[group_id]:
            raise ValueError(f"Duplicate qrel: {group_id}/{row['ayah']}")
        if row["grade"] != {"direct": 3, "partial": 2, "background": 1, "unrelated": 0}[row["label"]]:
            raise ValueError(f"Qrel label/grade mismatch: {group_id}/{row['ayah']}")
        by_group[group_id][row["ayah"]] = row["grade"]

    cases: list[dict[str, Any]] = []
    query_ids: set[str] = set()
    for question in sorted(questions, key=lambda row: row["group_id"]):
        group_id = question["group_id"]
        qrels = dict(sorted(by_group[group_id].items()))
        if not any(grade > 0 for grade in qrels.values()):
            raise ValueError(f"Group has no settled positive: {group_id}")
        judged = {
            verse["ayah"]
            for entries in question["evidence"].values()
            for verse in entries
        }
        for query in question["queries"]:
            query_id = query["query_id"]
            if query_id in query_ids:
                raise ValueError(f"Duplicate query ID: {query_id}")
            query_ids.add(query_id)
            cases.append(
                {
                    "id": query_id,
                    "group_id": group_id,
                    "query": query["text"],
                    "language": query["language"],
                    "best_settled_label": question["label_summary"]["best_settled_label"],
                    "compound_status": query["stats"]["compound_status"],
                    "qrels": qrels,
                    "disputed_ayahs": sorted(judged - set(qrels)),
                }
            )
    return cases


def fixture_bytes(cases: list[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(case, ensure_ascii=False, separators=(",", ":")) + "\n"
        for case in cases
    ).encode("utf8")


def prepare(args: argparse.Namespace) -> None:
    manifest = verify_v3_package(args.package)
    validator = subprocess.run(
        [sys.executable, str(args.package.parent / "validate_v3.py")],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf8",
    )
    cases = load_cases(args.package, args.split)
    corpus = load_corpus(args.corpus_db)
    if len(corpus) != 6236 or corpus_hash(corpus) != EXPECTED_CORPUS_SHA256:
        raise ValueError("Aayatun translation corpus does not match the frozen corpus")
    missing = {
        ayah
        for case in cases
        for ayah in set(case["qrels"]) | set(case["disputed_ayahs"])
        if ayah not in corpus
    }
    if missing:
        raise ValueError(f"Fixture references absent ayat: {sorted(missing)}")
    counts = {
        "query_cases": len(cases),
        "question_groups": len({case["group_id"] for case in cases}),
        "useful_answerable_cases": sum(
            any(grade >= 2 for grade in case["qrels"].values()) for case in cases
        ),
        "direct_answerable_cases": sum(
            case["best_settled_label"] == "direct" for case in cases
        ),
    }
    expected = (
        {
            "query_cases": 345,
            "question_groups": 341,
            "useful_answerable_cases": 294,
            "direct_answerable_cases": 54,
        }
        if args.split == "test"
        else {
            "query_cases": 260,
            "question_groups": 257,
            "useful_answerable_cases": 221,
            "direct_answerable_cases": 40,
        }
    )
    if counts != expected:
        raise ValueError(f"Frozen denominator mismatch: {counts} != {expected}")
    payload = fixture_bytes(cases)
    expected_hash = (
        EXPECTED_FIXTURE_SHA256 if args.split == "test" else EXPECTED_VALIDATION_FIXTURE_SHA256
    )
    if hashlib.sha256(payload).hexdigest() != expected_hash:
        raise ValueError(f"v3 {args.split} fixture differs from its frozen bytes")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fixture_path = args.output_dir / f"islamqa_v3_{args.split}_queries.jsonl"
    fixture_path.write_bytes(payload)
    fixture_manifest = {
        "profile": "pr2_frozen_v1",
        "dataset_version": manifest["dataset_version"],
        "dataset_commit": DATASET_COMMIT,
        "package_manifest_sha256": sha256_file(args.package / "manifest.json"),
        "validator": json.loads(validator.stdout),
        "split": args.split,
        "corpus": {"verses": len(corpus), "sha256": corpus_hash(corpus)},
        "counts": counts,
        "fixture_sha256": expected_hash,
        "success_definitions": {
            "relevant_hit@5": "At least one settled grade-1, grade-2, or grade-3 ayah in the first five.",
            "useful_hit@5": "At least one settled grade-2 or grade-3 ayah in the first five.",
            "direct_slice_useful_hit@5": f"Useful hit@5 among the {counts['direct_answerable_cases']} direct-answer wordings; not direct_hit@5.",
            "ndcg@10": "Gain 2^grade-1 with logarithmic rank discount, normalized by ideal qrels.",
        },
        "notes": [
            "The public test is a regression suite, not a blind holdout.",
            f"Qrels remain group-level; all {counts['query_cases']} original wordings are preserved.",
            "Confidence intervals resample question groups rather than wordings.",
        ],
    }
    write_json(args.output_dir / f"islamqa_v3_{args.split}_fixture_manifest.json", fixture_manifest)
    print(json.dumps(fixture_manifest, indent=2))


def load_fixture(output_dir: Path, split: str = "test") -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fixture_path = output_dir / f"islamqa_v3_{split}_queries.jsonl"
    manifest_path = output_dir / f"islamqa_v3_{split}_fixture_manifest.json"
    cases = read_jsonl(fixture_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    if sha256_file(fixture_path) != manifest["fixture_sha256"]:
        raise ValueError("Fixture hash mismatch")
    if len(cases) != manifest["counts"]["query_cases"]:
        raise ValueError("Fixture count mismatch")
    return cases, manifest


def scorer_module(package: Path) -> Callable[[list[str], dict[str, int]], dict[str, float]]:
    path = (package / "evaluate_search.py").resolve()
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("islamqa_frozen_scorer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the frozen scorer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.score_group


def validate_run(
    cases: list[dict[str, Any]],
    report: dict[str, Any],
    corpus_refs: set[str],
    max_results: int = 50,
) -> dict[str, list[str]]:
    expected = {case["id"] for case in cases}
    runs = report.get("runs")
    if not isinstance(runs, dict):
        raise ValueError("Ranking report runs must be an object")
    if set(runs) != expected:
        missing = sorted(expected - set(runs))
        extra = sorted(set(runs) - expected)
        raise ValueError(f"Ranking IDs mismatch: missing={missing}, extra={extra}")
    rankings: dict[str, list[str]] = {}
    for case_id, run in runs.items():
        if not isinstance(run, dict):
            raise ValueError(f"{case_id}: ranking entry must be an object")
        refs = run.get("refs")
        if not isinstance(refs, list) or len(refs) > max_results:
            raise ValueError(f"{case_id}: invalid ranked result list")
        if any(not isinstance(ref, str) or not AYAH_RE.fullmatch(ref) for ref in refs):
            raise ValueError(f"{case_id}: invalid ayah reference")
        if len(refs) != len(set(refs)):
            raise ValueError(f"{case_id}: duplicate ayah reference")
        try:
            elapsed = float(run["ms"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{case_id}: missing or invalid query latency") from exc
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError(f"{case_id}: query latency must be finite and non-negative")
        unknown = set(refs) - corpus_refs
        if unknown:
            raise ValueError(f"{case_id}: unknown ayat {sorted(unknown)}")
        rankings[case_id] = refs
    return rankings


def score_run(
    cases: list[dict[str, Any]],
    rankings: dict[str, list[str]],
    score_group: Callable[[list[str], dict[str, int]], dict[str, float]],
) -> dict[str, dict[str, float]]:
    scores: dict[str, dict[str, float]] = {}
    for case in cases:
        refs = rankings[case["id"]]
        row = score_group(refs, case["qrels"])
        useful_total = sum(grade >= 2 for grade in case["qrels"].values())
        useful_found = sum(case["qrels"].get(ref, 0) >= 2 for ref in refs[:50])
        row["useful_recall@50"] = (
            useful_found / useful_total if len(refs) >= 50 and useful_total else None
        )
        scores[case["id"]] = row
    return scores


def cluster_interval(
    cases: list[dict[str, Any]],
    values: dict[str, float | None],
    samples: int,
    seed: int,
) -> list[float] | None:
    if samples == 0:
        return None
    if any(values[case["id"]] is None for case in cases):
        return None
    by_group: dict[str, list[str]] = defaultdict(list)
    for case in cases:
        by_group[case["group_id"]].append(case["id"])
    group_ids = list(by_group)
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(samples):
        selected = [
            case_id
            for group_id in rng.choices(group_ids, k=len(group_ids))
            for case_id in by_group[group_id]
        ]
        draws.append(statistics.fmean(float(values[case_id]) for case_id in selected))
    ordered = sorted(draws)
    lower = ordered[math.floor(0.025 * (len(ordered) - 1))]
    upper = ordered[math.ceil(0.975 * (len(ordered) - 1))]
    return [round(lower, 6), round(upper, 6)]


def aggregate(
    cases: list[dict[str, Any]],
    scores: dict[str, dict[str, float]],
    samples: int,
    seed: int,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for metric in ("positive_hit@5", "useful_hit@5", "ndcg@10", "useful_recall@50"):
        values = {case["id"]: scores[case["id"]].get(metric) for case in cases}
        if any(value is None for value in values.values()):
            metrics[metric] = None
            continue
        mean = statistics.fmean(float(values[case["id"]]) for case in cases)
        metrics[metric] = {
            "mean": round(mean, 6),
            "ci95": cluster_interval(cases, values, samples, seed),
        }
    return {
        "queries": len(cases),
        "groups": len({case["group_id"] for case in cases}),
        "metrics": metrics,
    }


def paired_details(
    cases: list[dict[str, Any]],
    baseline: dict[str, dict[str, float]],
    candidate: dict[str, dict[str, float]],
    metric: str,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    deltas = {
        case["id"]: candidate[case["id"]][metric] - baseline[case["id"]][metric]
        for case in cases
    }
    wins = [case for case in cases if deltas[case["id"]] > 1e-12]
    losses = [case for case in cases if deltas[case["id"]] < -1e-12]
    ties = [case for case in cases if abs(deltas[case["id"]]) <= 1e-12]
    values = {case["id"]: deltas[case["id"]] for case in cases}
    return {
        "metric": metric,
        "wins": len(wins),
        "losses": len(losses),
        "ties": len(ties),
        "delta_mean": round(statistics.fmean(deltas.values()), 6),
        "delta_ci95": cluster_interval(cases, values, samples, seed),
        "gained_cases": [
            {"id": case["id"], "group_id": case["group_id"], "query": case["query"]}
            for case in wins
        ],
        "lost_cases": [
            {"id": case["id"], "group_id": case["group_id"], "query": case["query"]}
            for case in losses
        ],
    }


def rank_regression_details(
    cases: list[dict[str, Any]],
    baseline: dict[str, list[str]],
    candidate: dict[str, list[str]],
    minimum_grade: int,
    cutoff: int = 10,
) -> dict[str, Any]:
    def first_rank(refs: list[str], qrels: dict[str, int]) -> int:
        return next(
            (rank for rank, ref in enumerate(refs[:cutoff], start=1) if qrels.get(ref, 0) >= minimum_grade),
            cutoff + 1,
        )

    regressions = []
    for case in cases:
        baseline_rank = first_rank(baseline[case["id"]], case["qrels"])
        candidate_rank = first_rank(candidate[case["id"]], case["qrels"])
        if candidate_rank <= baseline_rank:
            continue
        baseline_ref = (
            baseline[case["id"]][baseline_rank - 1] if baseline_rank <= cutoff else None
        )
        candidate_ref = (
            candidate[case["id"]][candidate_rank - 1] if candidate_rank <= cutoff else None
        )
        regressions.append(
            {
                "id": case["id"],
                "group_id": case["group_id"],
                "query": case["query"],
                "baseline_rank": baseline_rank,
                "candidate_rank": candidate_rank,
                "baseline_ref": baseline_ref,
                "candidate_ref": candidate_ref,
            }
        )
    return {
        "cutoff": cutoff,
        "minimum_grade": minimum_grade,
        "count": len(regressions),
        "cases": regressions,
    }


def peak_rss_bytes() -> int:
    import psutil

    process = psutil.Process()
    try:
        info = process.memory_full_info()
    except psutil.AccessDenied:
        info = process.memory_info()
    return int(getattr(info, "peak_wset", info.rss))


def run_monitored_process(command: list[str]) -> int:
    import psutil

    process = psutil.Popen(command)
    peak = 0
    while True:
        try:
            members = [process, *process.children(recursive=True)]
            peak = max(peak, sum(member.memory_info().rss for member in members))
        except psutil.Error:
            pass
        if process.poll() is not None:
            break
        time.sleep(0.05)
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, command)
    return peak


def runtime_budgets(summary_runs: dict[str, Any]) -> dict[str, Any]:
    budgets = {}
    for name in summary_runs:
        run = summary_runs[name]
        p95 = run["timings_ms"]["p95"]
        index_bytes = run.get("index_bytes")
        rss_bytes = run.get("peak_rss_bytes")
        budgets[name] = {
            "p95_query_ms_max": math.ceil(round(p95 * 1.10, 6)),
            "index_bytes_max": math.ceil(round(index_bytes * 1.10, 6)) if index_bytes else None,
            "peak_rss_bytes_max": math.ceil(round(rss_bytes * 1.10, 6)) if rss_bytes else None,
        }
    return {
        "headroom": 1.10,
        "basis": "M00 measured runtime baseline plus 10%; fixed before M01 and not relaxed after test results.",
        "scope": "Query p95, serialized index/search database bytes, and peak process RSS (process-tree RSS for browser).",
        "runs": budgets,
    }


def result_coverage(
    cases: list[dict[str, Any]], rankings: dict[str, list[str]], cutoff: int = 10
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for case in cases:
        qrels = set(case["qrels"])
        disputed = set(case["disputed_ayahs"])
        for ayah in rankings[case["id"]][:cutoff]:
            counts["total_results"] += 1
            if ayah in qrels:
                counts["settled_judged_results"] += 1
            elif ayah in disputed:
                counts["disputed_results"] += 1
            else:
                counts["unjudged_results"] += 1
    return dict(counts)


def timing_summary(report: dict[str, Any], cases: list[dict[str, Any]]) -> dict[str, Any]:
    values = sorted(float(report["runs"][case["id"]]["ms"]) for case in cases)
    return {
        "p50": values[math.ceil(0.50 * len(values)) - 1],
        "p95": values[math.ceil(0.95 * len(values)) - 1],
        "queries": len(values),
    }


def format_metric(value: float | None, count: int | None, denominator: int) -> str:
    if value is None:
        return "—"
    if count is None:
        return f"{value:.4f}"
    return f"{count}/{denominator} ({value:.1%})"


def markdown_leaderboard(summary: dict[str, Any]) -> str:
    columns = list(RUN_COLUMNS)
    if "m01" in summary["runs"]:
        columns.append(("m01", summary["runs"]["m01"].get("leaderboard_label", "M01 candidate")))
    if "m02" in summary["runs"]:
        columns.append(("m02", summary["runs"]["m02"].get("leaderboard_label", "M02 candidate")))
    if "m03" in summary["runs"]:
        columns.append(("m03", summary["runs"]["m03"].get("leaderboard_label", "M03 candidate")))
    if "m04" in summary["runs"]:
        columns.append(("m04", summary["runs"]["m04"].get("leaderboard_label", "M04 candidate")))
    if "m05" in summary["runs"]:
        columns.append(("m05", summary["runs"]["m05"].get("leaderboard_label", "M05 candidate")))
    rows = (
        ("Relevant hit@5, all queries", "positive_hit@5", 345, "all"),
        ("Useful hit@5, answerable queries", "useful_hit@5", 294, "useful"),
        ("Useful hit@5, direct-answer queries", "useful_hit@5", 54, "direct"),
        ("nDCG@10, all queries", "ndcg@10", 345, "all"),
        ("Useful candidate recall@50", "useful_recall@50", 294, "useful"),
    )
    lines = [
        "| Measure | " + " | ".join(label for _, label in columns) + " |",
        "| --- | " + " | ".join("---:" for _ in columns) + " |",
    ]
    for label, metric, denominator, slice_name in rows:
        cells = []
        for key, _ in columns:
            run = summary["runs"][key]["headline"][slice_name]
            value = run["metrics"].get(metric)
            mean = value["mean"] if isinstance(value, dict) else value
            count = run["counts"].get(metric)
            cells.append(format_metric(mean, count, denominator))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def materialize_revision(repo: Path, revision: str, destination: Path) -> None:
    paths = [
        "build_search.py",
        "searchnorm.py",
        "searchtool.py",
        "db/quran.db",
        "db/en_quran.db",
        "db/en_kathir.db",
        "db/bn_zakaria.db",
        "db/ai_context.db",
        "db/quran_stat.json",
        "db/search_synonyms.json",
    ]
    archive = subprocess.check_output(
        ["git", "-C", str(repo), "archive", revision, *paths]
    )
    with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
        bundle.extractall(destination, filter="data")


def run_lexical_queries(args: argparse.Namespace) -> None:
    runtime = args.runtime
    norm_module = types.ModuleType("searchnorm")
    norm_module.__file__ = str(runtime / "searchnorm.py")
    exec(
        compile((runtime / "searchnorm.py").read_text(encoding="utf8"), "searchnorm.py", "exec"),
        norm_module.__dict__,
    )
    sys.modules["searchnorm"] = norm_module
    engine = types.ModuleType("islamqa_frozen_lexical")
    engine.__file__ = str(runtime / "searchtool.py")
    exec(
        compile((runtime / "searchtool.py").read_text(encoding="utf8"), "searchtool.py", "exec"),
        engine.__dict__,
    )
    if str(runtime) not in sys.path:
        sys.path.insert(0, str(runtime))
    engine._load_vocab()
    engine._load_bk_tree()
    engine.index_search("mercy", 10, field=args.field)
    cases = read_jsonl(runtime / FIXTURE_NAME)
    runs: dict[str, dict[str, Any]] = {}
    for index, case in enumerate(cases):
        started = time.perf_counter()
        hits, _ = engine.index_search(case["query"], 50, field=args.field)
        elapsed = round((time.perf_counter() - started) * 1000, 3)
        refs = [f"{hit['surah']}:{hit['verse']}" for hit in hits]
        if len(refs) != len(set(refs)):
            raise ValueError(f"Duplicate ayah returned for {case['id']}")
        source_fields = []
        for hit in hits:
            fields = [
                field
                for field in ("ayah", "translation", "en_tafsir", "bn_tafsir", "ai")
                if hit.get(field)
            ]
            source_fields.append(fields or ["unattributed_fallback"])
        runs[case["id"]] = {"refs": refs, "source_fields": source_fields, "ms": elapsed}
        if index % 50 == 0:
            print(f"{args.engine_name}: {index}/{len(cases)}", flush=True)
    write_json(runtime / "runs.json", runs)
    write_json(runtime / "runtime_metrics.json", {"peak_rss_bytes": peak_rss_bytes()})


def run_lexical(args: argparse.Namespace) -> None:
    cases, fixture_manifest = load_fixture(args.output_dir)
    revision = subprocess.check_output(
        ["git", "-C", str(args.repo), "rev-parse", args.revision],
        text=True,
    ).strip()
    with tempfile.TemporaryDirectory(prefix=".islamqa-frozen-", dir=WORKSPACE) as temp:
        runtime = Path(temp)
        materialize_revision(args.repo, revision, runtime)
        try:
            build = subprocess.run(
                [sys.executable, str(runtime / "build_search.py")],
                cwd=runtime,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf8",
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Lexical index build failed for {revision}\n{exc.stdout}\n{exc.stderr}"
            ) from exc
        print(build.stdout.strip().splitlines()[-1], flush=True)
        write_jsonl(runtime / FIXTURE_NAME, cases)
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "lexical-queries",
                "--runtime",
                str(runtime),
                "--engine-name",
                args.engine_name,
                "--field",
                args.field,
            ],
            cwd=runtime,
            check=True,
        )
        runs = json.loads((runtime / "runs.json").read_text(encoding="utf8"))
        runtime_metrics = json.loads((runtime / "runtime_metrics.json").read_text(encoding="utf8"))
        corpus = load_corpus(runtime / "db" / "en_quran.db")
        if corpus_hash(corpus) != fixture_manifest["corpus"]["sha256"]:
            raise ValueError("Lexical runtime corpus hash mismatch")
        report = {
            "engine": f"Aayatun {args.engine_name}",
            "profile": "pr2_frozen_v1",
            "revision": revision,
            "field": args.field,
            "source_attribution": args.field,
            "fixture_sha256": fixture_manifest["fixture_sha256"],
            "corpus": {"verses": len(corpus), "sha256": corpus_hash(corpus)},
            "search_db_sha256": sha256_file(runtime / "db" / "search.db"),
            "search_db_bytes": (runtime / "db" / "search.db").stat().st_size,
            "peak_rss_bytes": runtime_metrics["peak_rss_bytes"],
            "runs": runs,
        }
    write_json(args.output, report)
    print(json.dumps({"output": str(args.output), "queries": len(runs)}, indent=2))


def model_snapshot() -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            E5_MODEL,
            revision=E5_REVISION,
            local_files_only=True,
        )
    )


def run_dense(args: argparse.Namespace) -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    from utils import VectorSearchDeployment

    cases, fixture_manifest = load_fixture(args.output_dir)
    corpus = load_corpus(args.corpus_db)
    if len(corpus) != 6236 or corpus_hash(corpus) != fixture_manifest["corpus"]["sha256"]:
        raise ValueError("Dense corpus does not match the frozen fixture")
    snapshot = model_snapshot()
    embeddings = HuggingFaceEmbeddings(
        model_name=str(snapshot),
        model_kwargs={"device": "cpu", "trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True, "max_seq_length": 512},
    )
    documents = [
        Document(
            page_content="passage: " + text,
            metadata={
                "surah": int(ref.split(":")[0]),
                "ayah": int(ref.split(":")[1]),
                "field": "translation",
                "source": "saheeh_international",
                "chunk": 0,
            },
        )
        for ref, text in sorted(corpus.items(), key=lambda item: tuple(map(int, item[0].split(":"))))
    ]
    index_path = args.runtime_dir / "e5-small" / "quran"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    database = FAISS.from_documents(
        documents,
        embeddings,
        normalize_L2=True,
        distance_strategy="MAX_INNER_PRODUCT",
    )
    database.save_local(str(index_path))
    deployment = VectorSearchDeployment(str(index_path), str(snapshot), "cpu")
    runs: dict[str, dict[str, Any]] = {}
    for index, case in enumerate(cases):
        started = time.perf_counter()
        results = deployment.search(
            "best_match",
            "query: " + case["query"],
            k=50,
            retrieval_mode="similarity",
            fetch_k=50,
            mmr_lambda=1.0,
        )
        elapsed = round((time.perf_counter() - started) * 1000, 3)
        refs = [
            f"{doc.metadata['surah']}:{doc.metadata['ayah']}" for doc, _ in results
        ]
        if len(refs) != len(set(refs)):
            raise ValueError(f"Dense retrieval returned duplicate ayah for {case['id']}")
        runs[case["id"]] = {"refs": refs, "ms": elapsed}
        if index % 50 == 0:
            print(f"offline dense: {index}/{len(cases)}", flush=True)
    model_files = {
        name: sha256_file(snapshot / name)
        for name in (
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "model.safetensors",
        )
    }
    report = {
        "engine": "Offline multilingual E5-small float32 FAISS",
        "profile": "pr2_frozen_v1",
        "model": E5_MODEL,
        "model_revision": E5_REVISION,
        "model_files": model_files,
        "dtype": "float32",
        "distance": "max_inner_product",
        "fetch_k": 50,
        "field": "translation",
        "source_attribution": "translation",
        "fixture_sha256": fixture_manifest["fixture_sha256"],
        "corpus": {"verses": len(corpus), "sha256": corpus_hash(corpus)},
        "index_bytes": sum(path.stat().st_size for path in index_path.rglob("*") if path.is_file()),
        "peak_rss_bytes": peak_rss_bytes(),
        "runs": runs,
    }
    write_json(args.output, report)
    print(json.dumps({"output": str(args.output), "queries": len(runs)}, indent=2))


def load_or_build_dense_index(args: argparse.Namespace, cases: list[dict[str, Any]], fixture_manifest: dict[str, Any]):
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    from utils import VectorSearchDeployment

    corpus = load_corpus(args.corpus_db)
    if len(corpus) != 6236 or corpus_hash(corpus) != fixture_manifest["corpus"]["sha256"]:
        raise ValueError("Dense corpus does not match the frozen fixture")
    snapshot = model_snapshot()
    embeddings = HuggingFaceEmbeddings(
        model_name=str(snapshot),
        model_kwargs={"device": "cpu", "trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True, "max_seq_length": 512},
    )
    index_path = args.runtime_dir / "e5-small" / "quran"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    if not (index_path / "index.faiss").exists():
        documents = [
            Document(
                page_content="passage: " + text,
                metadata={
                    "surah": int(ref.split(":")[0]),
                    "ayah": int(ref.split(":")[1]),
                    "field": "translation",
                    "source": "saheeh_international",
                    "chunk": 0,
                },
            )
            for ref, text in sorted(corpus.items(), key=lambda item: tuple(map(int, item[0].split(":"))))
        ]
        database = FAISS.from_documents(
            documents,
            embeddings,
            normalize_L2=True,
            distance_strategy="MAX_INNER_PRODUCT",
        )
        database.save_local(str(index_path))
    deployment = VectorSearchDeployment(str(index_path), str(snapshot), "cpu")
    if deployment.db.index.ntotal != len(corpus):
        raise ValueError("Dense index size does not match the frozen corpus")
    return deployment, index_path, snapshot


def dense_model_files(snapshot: Path) -> dict[str, str]:
    return {
        name: sha256_file(snapshot / name)
        for name in (
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "model.safetensors",
        )
    }


def canonical_config(config: dict[str, Any]) -> dict[str, Any]:
    mode = config["retrieval_mode"]
    raw_fetch_k = config["fetch_k"]
    if isinstance(raw_fetch_k, bool):
        raise ValueError("fetch_k must be 20, 50, or 100")
    fetch_k = int(raw_fetch_k)
    if isinstance(raw_fetch_k, float) and raw_fetch_k != fetch_k:
        raise ValueError("fetch_k must be 20, 50, or 100")
    relevance_weight = float(config["mmr_lambda"])
    if mode not in ("similarity", "mmr"):
        raise ValueError("retrieval_mode must be similarity or mmr")
    if fetch_k not in (20, 50, 100):
        raise ValueError("fetch_k must be 20, 50, or 100")
    if relevance_weight not in (1.0, 0.85, 0.7):
        raise ValueError("mmr_lambda must be 1.0, 0.85, or 0.7")
    return {
        "retrieval_mode": mode,
        "fetch_k": fetch_k,
        "mmr_lambda": relevance_weight,
    }


def config_sha256(config: dict[str, Any]) -> str:
    payload = json.dumps(canonical_config(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf8")).hexdigest()


def execute_dense_config(
    deployment: Any,
    cases: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    config = canonical_config(config)
    result_k = min(50, config["fetch_k"])
    runs: dict[str, dict[str, Any]] = {}
    for index, case in enumerate(cases):
        started = time.perf_counter()
        results = deployment.search(
            "best_match",
            "query: " + case["query"],
            k=result_k,
            retrieval_mode=config["retrieval_mode"],
            fetch_k=config["fetch_k"],
            mmr_lambda=config["mmr_lambda"],
        )
        elapsed = round((time.perf_counter() - started) * 1000, 3)
        refs = [f"{doc.metadata['surah']}:{doc.metadata['ayah']}" for doc, _ in results]
        if len(refs) != len(set(refs)):
            raise ValueError(f"Dense retrieval returned duplicate ayah for {case['id']}")
        if len(refs) > config["fetch_k"] or len(refs) > result_k:
            raise ValueError(f"Dense retrieval exceeded its depth bound for {case['id']}")
        runs[case["id"]] = {
            "refs": refs,
            "source_fields": [["translation"] for _ in refs],
            "ms": elapsed,
        }
        if index % 50 == 0:
            print(f"{config['retrieval_mode']}/{config['fetch_k']}/{config['mmr_lambda']}: {index}/{len(cases)}", flush=True)
    return {"config": config, "result_k": result_k, "runs": runs}


def dense_config_report(
    args: argparse.Namespace,
    cases: list[dict[str, Any]],
    fixture_manifest: dict[str, Any],
    deployment: Any,
    index_path: Path,
    snapshot: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    execution = execute_dense_config(deployment, cases, config)
    report = {
        "engine": "M01 offline multilingual E5-small float32 FAISS",
        "profile": "pr2_frozen_v1",
        "model": E5_MODEL,
        "model_revision": E5_REVISION,
        "model_files": dense_model_files(snapshot),
        "dtype": "float32",
        "distance": "max_inner_product",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
        "field": "translation",
        "source_attribution": "translation",
        "fixture_sha256": fixture_manifest["fixture_sha256"],
        "corpus": {"verses": len(load_corpus(args.corpus_db)), "sha256": fixture_manifest["corpus"]["sha256"]},
        "index_bytes": sum(path.stat().st_size for path in index_path.rglob("*") if path.is_file()),
        "peak_rss_bytes": peak_rss_bytes(),
        **execution,
    }
    report["config_sha256"] = config_sha256(report["config"])
    return report


def run_dense_config(args: argparse.Namespace) -> None:
    cases, fixture_manifest = load_fixture(args.output_dir, args.split)
    config = canonical_config(
        {
            "retrieval_mode": args.retrieval_mode,
            "fetch_k": args.fetch_k,
            "mmr_lambda": args.mmr_lambda,
        }
    )
    if args.selection:
        selection = json.loads(args.selection.read_text(encoding="utf8"))
        if selection.get("split") != "validation":
            raise ValueError("M01 selection must be frozen on validation")
        if canonical_config(selection["selected_config"]) != config:
            raise ValueError("Requested test configuration does not match the frozen selection")
        for relative, digest in selection["code_sha256"].items():
            if sha256_file(ROOT / relative) != digest:
                raise ValueError(f"Frozen M01 code changed after selection: {relative}")
    deployment, index_path, snapshot = load_or_build_dense_index(args, cases, fixture_manifest)
    report = dense_config_report(
        args, cases, fixture_manifest, deployment, index_path, snapshot, config
    )
    if args.selection:
        report["selection_sha256"] = sha256_file(args.selection)
        report["selection"] = selection
    write_json(args.output, report)
    print(json.dumps({"output": str(args.output), "config": config, "queries": len(cases)}, indent=2))


def validation_summary(
    name: str,
    report: dict[str, Any],
    cases: list[dict[str, Any]],
    corpus_refs: set[str],
    fixture_manifest: dict[str, Any],
    score_group: Callable[[list[str], dict[str, int]], dict[str, float]],
    bootstrap: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, list[str]], dict[str, dict[str, float]]]:
    rankings = validate_run(cases, report, corpus_refs)
    scores = score_run(cases, rankings, score_group)
    useful = [case for case in cases if any(grade >= 2 for grade in case["qrels"].values())]
    direct = [case for case in cases if case["best_settled_label"] == "direct"]
    headline = {}
    for slice_name, selected in (("all", cases), ("useful", useful), ("direct", direct)):
        row = aggregate(selected, scores, bootstrap, seed)
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
        "timings_ms": timing_summary(report, cases),
        "coverage@10": result_coverage(cases, rankings, 10),
        "index_bytes": report["index_bytes"],
        "peak_rss_bytes": report["peak_rss_bytes"],
    }
    return summary, rankings, scores


def validation_gate(parent: dict[str, Any], candidate: dict[str, Any]) -> bool:
    parent_useful = parent["headline"]["useful"]["counts"]["useful_hit@5"]
    candidate_useful = candidate["headline"]["useful"]["counts"]["useful_hit@5"]
    parent_ndcg = parent["headline"]["all"]["metrics"]["ndcg@10"]["mean"]
    candidate_ndcg = candidate["headline"]["all"]["metrics"]["ndcg@10"]["mean"]
    return candidate_useful > parent_useful and candidate_ndcg >= parent_ndcg


def choose_validation_candidate(parent: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [candidate for candidate in candidates if validation_gate(parent, candidate)]
    if not eligible:
        return parent
    return max(
        eligible,
        key=lambda candidate: (
            candidate["headline"]["useful"]["counts"]["useful_hit@5"],
            candidate["headline"]["all"]["metrics"]["ndcg@10"]["mean"],
            candidate["headline"]["all"]["counts"]["positive_hit@5"],
        ),
    )


def run_dense_sweep(args: argparse.Namespace) -> None:
    from datetime import datetime, timezone

    cases, fixture_manifest = load_fixture(args.output_dir, "validation")
    corpus = load_corpus(args.corpus_db)
    corpus_refs = set(corpus)
    score_group = scorer_module(args.scorer_package)
    deployment, index_path, snapshot = load_or_build_dense_index(args, cases, fixture_manifest)
    initial_configs = (
        ("similarity_fetch20", {"retrieval_mode": "similarity", "fetch_k": 20, "mmr_lambda": 1.0}),
        ("similarity_fetch50_parent", {"retrieval_mode": "similarity", "fetch_k": 50, "mmr_lambda": 1.0}),
        ("similarity_fetch100", {"retrieval_mode": "similarity", "fetch_k": 100, "mmr_lambda": 1.0}),
        ("mmr_fetch50_lambda1", {"retrieval_mode": "mmr", "fetch_k": 50, "mmr_lambda": 1.0}),
        ("mmr_fetch50_lambda085", {"retrieval_mode": "mmr", "fetch_k": 50, "mmr_lambda": 0.85}),
        ("mmr_fetch50_lambda07", {"retrieval_mode": "mmr", "fetch_k": 50, "mmr_lambda": 0.7}),
    )
    summaries: dict[str, dict[str, Any]] = {}
    scores_by_name: dict[str, dict[str, dict[str, float]]] = {}
    artifacts: dict[str, dict[str, Any]] = {}

    def run_configs(configs):
        for name, config in configs:
            report = dense_config_report(
                args, cases, fixture_manifest, deployment, index_path, snapshot, config
            )
            path = args.output_dir / "validation_runs" / f"{name}.json"
            write_json(path, report)
            summary, _, scores = validation_summary(
                name, report, cases, corpus_refs, fixture_manifest, score_group,
                args.bootstrap, args.seed,
            )
            summaries[name] = summary
            scores_by_name[name] = scores
            artifacts[name] = {"file": str(path.relative_to(args.output_dir)), "sha256": sha256_file(path)}

    run_configs(initial_configs)
    parent = summaries["similarity_fetch50_parent"]
    initial_candidates = [summaries[name] for name, _ in initial_configs if name != parent["name"]]
    selected_before_depth = choose_validation_candidate(parent, initial_candidates)
    stages = [{
        "name": "parent_and_one_knob_ablations",
        "candidates": [parent["name"], *[name for name, _ in initial_configs if name != parent["name"]]],
        "selected": selected_before_depth["name"],
    }]

    selected = selected_before_depth
    selected_config = canonical_config(selected["config"])
    if selected_config["retrieval_mode"] == "mmr" and selected_config["mmr_lambda"] < 1.0:
        suffix = str(selected_config["mmr_lambda"]).replace(".", "")
        followups = (
            (f"mmr_fetch20_lambda{suffix}", {**selected_config, "fetch_k": 20}),
            (f"mmr_fetch100_lambda{suffix}", {**selected_config, "fetch_k": 100}),
        )
        run_configs(followups)
        followup_candidates = [summaries[name] for name, _ in followups]
        selected = choose_validation_candidate(selected_before_depth, followup_candidates)
        stages.append({
            "name": "candidate_depth_at_selected_lambda",
            "parent": selected_before_depth["name"],
            "candidates": [name for name, _ in followups],
            "selected": selected["name"],
        })

    useful = [case for case in cases if any(grade >= 2 for grade in case["qrels"].values())]
    marginal_effects = {}
    for name, summary in summaries.items():
        marginal_effects[name] = {
            "vs_parent": {
                "positive_hit@5": paired_details(
                    cases, scores_by_name[parent["name"]], scores_by_name[name],
                    "positive_hit@5", args.bootstrap, args.seed,
                ),
                "useful_hit@5": paired_details(
                    useful, scores_by_name[parent["name"]], scores_by_name[name],
                    "useful_hit@5", args.bootstrap, args.seed,
                ),
                "ndcg@10": paired_details(
                    cases, scores_by_name[parent["name"]], scores_by_name[name],
                    "ndcg@10", args.bootstrap, args.seed,
                ),
            }
        }
    repeat = execute_dense_config(deployment, cases, selected["config"])
    repeat_rankings = {case_id: run["refs"] for case_id, run in repeat["runs"].items()}
    selected_report_path = args.output_dir / "validation_runs" / f"{selected['name']}.json"
    selected_report = json.loads(selected_report_path.read_text(encoding="utf8"))
    selected_rankings = {case_id: run["refs"] for case_id, run in selected_report["runs"].items()}
    determinism_exact = sum(
        repeat_rankings[case["id"]] == selected_rankings[case["id"]] for case in cases
    )
    if determinism_exact != len(cases):
        raise ValueError("Selected retrieval configuration is not deterministic on validation")
    selected_changed = selected["config_sha256"] != parent["config_sha256"]
    selection = {
        "profile": "pr2_frozen_v1",
        "experiment": "M01",
        "split": "validation",
        "dataset": fixture_manifest,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_accessed": False,
        "selection_policy": "Higher validation useful hit@5 with no validation nDCG@10 decrease; keep the simpler parent when no candidate passes. Depth follow-ups vary only FETCH_K at the selected lambda.",
        "stages": stages,
        "candidates": summaries,
        "marginal_effects": marginal_effects,
        "selected_name": selected["name"],
        "selected_config": canonical_config(selected["config"]),
        "selected_config_sha256": selected["config_sha256"],
        "selected_changed": selected_changed,
        "validation_gate_passed": selected_changed,
        "determinism_check": {
            "method": "Repeat the complete selected validation ranking in the same process and compare every ayah ID, including any score ties.",
            "queries": len(cases),
            "exact_queries": determinism_exact,
            "passed": determinism_exact == len(cases),
        },
        "model": {"name": E5_MODEL, "revision": E5_REVISION, "files": dense_model_files(snapshot)},
        "prefixes": {"query": "query: ", "passage": "passage: "},
        "code_sha256": {
            "evaluation/islamqa_frozen.py": sha256_file(HERE / "islamqa_frozen.py"),
            "utils.py": sha256_file(ROOT / "utils.py"),
            "evaluation/test_retrieval_config.py": sha256_file(HERE / "test_retrieval_config.py"),
        },
        "validation_ranking_artifacts": artifacts,
    }
    write_json(args.selection, selection)
    rows = []
    for name, summary in summaries.items():
        rows.append({
            "name": name,
            **summary["config"],
            "relevant_hit@5": summary["headline"]["all"]["counts"]["positive_hit@5"],
            "useful_hit@5": summary["headline"]["useful"]["counts"]["useful_hit@5"],
            "direct_useful_hit@5": summary["headline"]["direct"]["counts"]["useful_hit@5"],
            "ndcg@10": summary["headline"]["all"]["metrics"]["ndcg@10"]["mean"],
            "p95_ms": summary["timings_ms"]["p95"],
        })
    print(json.dumps({"selected": selection["selected_config"], "rows": rows}, indent=2))


def run_browser(args: argparse.Namespace) -> None:
    harness = args.pr2_eval_repo / "evaluation" / "islamqa_pr2_browser.cjs"
    command = [
        "node",
        str(harness),
        "--root",
        str(args.pr2_eval_repo),
        "--model-root",
        str(args.aayatun_repo / "static" / "meaning" / "models"),
        "--fixture",
        str(args.output_dir / FIXTURE_NAME),
        "--manifest",
        str(args.output_dir / MANIFEST_NAME),
        "--output",
        str(args.output),
        "--attempts",
        "1",
    ]
    peak_rss = run_monitored_process(command)
    report = json.loads(args.output.read_text(encoding="utf8"))
    if report.get("product_commit") != PR2_REVISION:
        raise ValueError("Browser reproduction did not use the pinned PR #2 product commit")
    index_path = args.pr2_eval_repo / "static" / "meaning" / "index.json"
    index = json.loads(index_path.read_text(encoding="utf8"))
    report.update(
        {
            "profile": "pr2_frozen_v1",
            "field": "translation",
            "source_attribution": "translation",
            "index_bytes": (args.pr2_eval_repo / "static" / "meaning" / index["file"]).stat().st_size,
            "peak_rss_bytes": peak_rss,
        }
    )
    write_json(args.output, report)
    print(json.dumps({"output": str(args.output)}, indent=2))


def load_run(
    path: Path,
    cases: list[dict[str, Any]],
    corpus_refs: set[str],
    fixture_manifest: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    report = json.loads(path.read_text(encoding="utf8"))
    if report.get("profile") != "pr2_frozen_v1":
        raise ValueError(f"Evaluation profile mismatch: {path}")
    if report.get("fixture_sha256") != fixture_manifest["fixture_sha256"]:
        raise ValueError(f"Fixture hash mismatch: {path}")
    corpus = report.get("corpus")
    digest = corpus.get("sha256") if isinstance(corpus, dict) else corpus
    if digest != fixture_manifest["corpus"]["sha256"]:
        raise ValueError(f"Corpus hash mismatch: {path}")
    rankings = validate_run(cases, report, corpus_refs)
    return report, rankings


def build_report(args: argparse.Namespace) -> None:
    cases, fixture_manifest = load_fixture(args.output_dir)
    corpus = load_corpus(args.corpus_db)
    corpus_refs = set(corpus)
    score_group = scorer_module(args.scorer_package)
    paths = {
        "deploy": args.output_dir / "deploy_rankings.json",
        "fcs_translation": args.output_dir / "fcs_translation_rankings.json",
        "fcs_all": args.output_dir / "fcs_all_rankings.json",
        "browser": args.output_dir / "browser_rankings.json",
        "dense": args.output_dir / "dense_rankings.json",
    }
    reports: dict[str, dict[str, Any]] = {}
    rankings: dict[str, dict[str, list[str]]] = {}
    scores: dict[str, dict[str, dict[str, float]]] = {}
    expected_fields = {
        "deploy": "translation",
        "fcs_translation": "translation",
        "fcs_all": "all",
        "browser": "translation",
        "dense": "translation",
    }
    for name, path in paths.items():
        reports[name], rankings[name] = load_run(
            path, cases, corpus_refs, fixture_manifest
        )
        if reports[name].get("field") != expected_fields[name]:
            raise ValueError(f"{name}: expected {expected_fields[name]} field scope")
        scores[name] = score_run(cases, rankings[name], score_group)

    useful = [case for case in cases if any(grade >= 2 for grade in case["qrels"].values())]
    direct = [case for case in cases if case["best_settled_label"] == "direct"]
    slices = {
        "all": cases,
        "useful": useful,
        "direct": direct,
        "partial": [case for case in cases if case["best_settled_label"] == "partial"],
        "background": [case for case in cases if case["best_settled_label"] == "background"],
        "long": [case for case in cases if len(case["query"]) >= 120],
        "compound": [case for case in cases if case["compound_status"] == "review_required"],
    }
    summary_runs: dict[str, Any] = {}
    for name in paths:
        headline = {}
        for slice_name, selected in slices.items():
            aggregate_row = aggregate(selected, scores[name], args.bootstrap, args.seed)
            counts = {}
            for metric in ("positive_hit@5", "useful_hit@5"):
                counts[metric] = sum(
                    scores[name][case["id"]][metric] > 0 for case in selected
                )
            headline[slice_name] = {**aggregate_row, "counts": counts}
        summary_runs[name] = {
            "engine": reports[name]["engine"],
            "revision": reports[name].get("revision") or reports[name].get("product_commit"),
            "field": reports[name].get("field"),
            "source_attribution": reports[name].get("source_attribution"),
            "headline": headline,
            "timings_ms": timing_summary(reports[name], cases),
            "coverage@10": result_coverage(cases, rankings[name], 10),
            "coverage@50": result_coverage(cases, rankings[name], 50),
            "index_bytes": reports[name].get("index_bytes") or reports[name].get("search_db_bytes"),
            "peak_rss_bytes": reports[name].get("peak_rss_bytes"),
            "provenance": {key: value for key, value in reports[name].items() if key != "runs"},
        }

    comparisons = {}
    for baseline, candidate in (
        ("deploy", "fcs_translation"),
        ("fcs_translation", "fcs_all"),
        ("fcs_translation", "browser"),
        ("browser", "dense"),
    ):
        comparisons[f"{candidate}_vs_{baseline}"] = {
            "positive_hit@5": paired_details(
                cases, scores[baseline], scores[candidate], "positive_hit@5",
                args.bootstrap, args.seed,
            ),
            "useful_hit@5": paired_details(
                useful, scores[baseline], scores[candidate], "useful_hit@5",
                args.bootstrap, args.seed,
            ),
            "ndcg@10": paired_details(
                cases, scores[baseline], scores[candidate], "ndcg@10",
                args.bootstrap, args.seed,
            ),
            "positive_rank_regressions@10": rank_regression_details(
                cases, rankings[baseline], rankings[candidate], minimum_grade=1
            ),
            "useful_rank_regressions@10": rank_regression_details(
                useful, rankings[baseline], rankings[candidate], minimum_grade=2
            ),
        }

    browser_reference = json.loads(args.browser_reference.read_text(encoding="utf8"))
    reference_corpus = browser_reference.get("corpus")
    reference_corpus_hash = (
        reference_corpus.get("sha256") if isinstance(reference_corpus, dict) else reference_corpus
    )
    if browser_reference.get("fixture_sha256") != fixture_manifest["fixture_sha256"]:
        raise ValueError("Browser reference fixture hash mismatch")
    if reference_corpus_hash != fixture_manifest["corpus"]["sha256"]:
        raise ValueError("Browser reference corpus hash mismatch")
    reference_rankings = validate_run(cases, browser_reference, corpus_refs, max_results=10)
    exact = sum(
        reference_rankings[case["id"]] == rankings["browser"][case["id"]]
        for case in cases
    )
    overlap = sum(
        len(set(reference_rankings[case["id"]]) & set(rankings["browser"][case["id"]]))
        for case in cases
    )
    dense_exact = sum(
        rankings["browser"][case["id"]][:10] == rankings["dense"][case["id"]][:10]
        for case in cases
    )
    dense_overlap = sum(
        len(set(rankings["browser"][case["id"]]) & set(rankings["dense"][case["id"]][:10]))
        for case in cases
    )
    baseline_counts = {
        "relevant": summary_runs["browser"]["headline"]["all"]["counts"]["positive_hit@5"],
        "useful": summary_runs["browser"]["headline"]["useful"]["counts"]["useful_hit@5"],
        "direct_useful": summary_runs["browser"]["headline"]["direct"]["counts"]["useful_hit@5"],
    }
    accepted = (
        baseline_counts == {"relevant": 46, "useful": 44, "direct_useful": 16}
        and exact == 345
    )
    report = {
        "profile": "pr2_frozen_v1",
        "outcome": "accepted" if accepted else "rejected",
        "contribution": "Frozen v3 query-level contract and reproducible baselines; no relevance gain claimed.",
        "dataset": fixture_manifest,
        "scorer_sha256": text_sha256(args.scorer_package / "evaluate_search.py"),
        "code_sha256": {
            "evaluation/islamqa_frozen.py": sha256_file(HERE / "islamqa_frozen.py"),
            "run_benchmark.py": sha256_file(ROOT / "run_benchmark.py"),
            "utils.py": sha256_file(ROOT / "utils.py"),
        },
        "slice_definitions": {
            "all": "All 345 original wordings.",
            "useful": "Wordings with at least one settled grade-2 or grade-3 qrel; 294 cases.",
            "direct": "best_settled_label is direct; 54 wordings.",
            "partial": "best_settled_label is partial.",
            "background": "best_settled_label is background.",
            "long": "Query text has at least 120 characters.",
            "compound": "compound_status is review_required.",
        },
        "promotion_policy": {
            "selection_split": "validation",
            "test_role": "Frozen public regression suite; never blind and never used for tuning.",
            "validation_gate": "Higher useful hit@5 with no nDCG@10 decrease versus the accepted parent.",
            "test_gate": "No decrease in relevant hit@5, useful hit@5, direct-slice useful hit@5, or nDCG@10 versus the accepted parent.",
            "candidate_grids": "M00 has no relevance grid. Each later issue must record its validation grid before test access.",
            "rollback": "Disable the issue's configuration switch or restore the recorded parent commit/configuration.",
        },
        "runtime_budgets": runtime_budgets(summary_runs),
        "profile_ablation": {
            "legacy_entrypoint": "run_benchmark.py without EVAL_PROFILE retains its existing path.",
            "frozen_entrypoint": "EVAL_PROFILE=pr2_frozen_v1 routes to evaluation/islamqa_frozen.py before legacy imports.",
            "retrieval_algorithm_changed": False,
            "parity_evidence": "Exact reproduction of all pinned PR #2 browser rankings.",
        },
        "runs": summary_runs,
        "source_ranking_artifacts": {
            name: {
                "file": path.name,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for name, path in paths.items()
        },
        "comparisons": comparisons,
        "browser_reproduction": {
            "reference_sha256": sha256_file(args.browser_reference),
            "exact_queries": exact,
            "total_queries": len(cases),
            "top10_overlap": overlap,
            "max_top10_overlap": len(cases) * 10,
        },
        "backend_parity": {
            "browser_vs_dense_exact_top10_queries": dense_exact,
            "browser_vs_dense_top10_overlap": dense_overlap,
            "max_top10_overlap": len(cases) * 10,
            "note": "Browser int8 document vectors and float32 FAISS are separate backends; differences are reported, not silently treated as parity.",
        },
        "acceptance": {
            "accepted": accepted,
            "browser_headline_counts": baseline_counts,
            "required_browser_counts": {"relevant": 46, "useful": 44, "direct_useful": 16},
            "browser_reproduction_exact": exact == len(cases),
        },
        "limitations": fixture_manifest["notes"],
    }
    table = markdown_leaderboard(report)
    (args.output_dir / "leaderboard.md").write_text(table + "\n", encoding="utf8")
    for name, selected_rankings in rankings.items():
        write_jsonl(
            args.output_dir / f"{name}_query_rankings.jsonl",
            (
                {
                    "id": case["id"],
                    "group_id": case["group_id"],
                    "query": case["query"],
                    "ranking": selected_rankings[case["id"]],
                    "source_fields": reports[name]["runs"][case["id"]].get(
                        "source_fields",
                        [reports[name]["field"]] * len(selected_rankings[case["id"]]),
                    ),
                }
                for case in cases
            ),
        )
    report["query_ranking_artifacts"] = {
        name: {
            "file": f"{name}_query_rankings.jsonl",
            "sha256": sha256_file(args.output_dir / f"{name}_query_rankings.jsonl"),
            "bytes": (args.output_dir / f"{name}_query_rankings.jsonl").stat().st_size,
        }
        for name in paths
    }
    write_json(args.output_dir / "report.json", report)
    print(table)
    print()
    print(json.dumps(report["acceptance"], indent=2))


def load_committed_rankings(
    name: str,
    baseline_dir: Path,
    baseline_report: dict[str, Any],
    cases: list[dict[str, Any]],
    corpus_refs: set[str],
) -> tuple[dict[str, list[str]], dict[str, dict[str, Any]]]:
    artifact = baseline_report["query_ranking_artifacts"][name]
    path = baseline_dir / artifact["file"]
    if sha256_file(path) != artifact["sha256"]:
        raise ValueError(f"Committed baseline ranking hash mismatch: {name}")
    rows = read_jsonl(path)
    expected = {case["id"] for case in cases}
    if len(rows) != len(expected) or {row["id"] for row in rows} != expected:
        raise ValueError(f"Committed baseline query IDs mismatch: {name}")
    rankings: dict[str, list[str]] = {}
    runs: dict[str, dict[str, Any]] = {}
    for row in rows:
        refs = row["ranking"]
        if not isinstance(refs, list) or len(refs) > 50:
            raise ValueError(f"Invalid committed baseline ranking: {name}/{row['id']}")
        if any(not isinstance(ref, str) or not AYAH_RE.fullmatch(ref) for ref in refs):
            raise ValueError(f"Invalid committed baseline ayah: {name}/{row['id']}")
        if len(refs) != len(set(refs)) or set(refs) - corpus_refs:
            raise ValueError(f"Duplicate or unknown committed baseline ayah: {name}/{row['id']}")
        source_fields = row.get("source_fields", [[baseline_report["runs"][name]["field"]] for _ in refs])
        if len(source_fields) != len(refs):
            raise ValueError(f"Baseline source attribution mismatch: {name}/{row['id']}")
        rankings[row["id"]] = refs
        runs[row["id"]] = {"refs": refs, "source_fields": source_fields}
    return rankings, runs


def candidate_headline_summary(
    report: dict[str, Any],
    cases: list[dict[str, Any]],
    rankings: dict[str, list[str]],
    scores: dict[str, dict[str, float]],
    bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    useful = [case for case in cases if any(grade >= 2 for grade in case["qrels"].values())]
    direct = [case for case in cases if case["best_settled_label"] == "direct"]
    slices = {
        "all": cases,
        "useful": useful,
        "direct": direct,
        "partial": [case for case in cases if case["best_settled_label"] == "partial"],
        "background": [case for case in cases if case["best_settled_label"] == "background"],
        "long": [case for case in cases if len(case["query"]) >= 120],
        "compound": [case for case in cases if case["compound_status"] == "review_required"],
    }
    headline = {}
    for slice_name, selected in slices.items():
        row = aggregate(selected, scores, bootstrap, seed)
        row["counts"] = {
            metric: sum(scores[case["id"]][metric] > 0 for case in selected)
            for metric in ("positive_hit@5", "useful_hit@5")
        }
        headline[slice_name] = row
    config = report["config"]
    if report.get("experiment") == "M03":
        leaderboard_label = f"M03 {config['representation_scope']}"
    elif "representation_scope" in config:
        leaderboard_label = f"M02 {config['representation_scope']}"
    else:
        leaderboard_label = f"M01 {config['retrieval_mode']} λ={config['mmr_lambda']:g} k={config['fetch_k']}"
    return {
        "engine": report["engine"],
        "leaderboard_label": leaderboard_label,
        "revision": report.get("model_revision"),
        "field": report["field"],
        "source_attribution": report["source_attribution"],
        "headline": headline,
        "timings_ms": timing_summary(report, cases),
        "coverage@10": result_coverage(cases, rankings, 10),
        "coverage@50": result_coverage(cases, rankings, 50),
        "index_bytes": report["index_bytes"],
        "peak_rss_bytes": report["peak_rss_bytes"],
        "provenance": {key: value for key, value in report.items() if key != "runs"},
    }


def headline_test_counts(summary: dict[str, Any]) -> dict[str, float]:
    return {
        "relevant_hit@5": summary["headline"]["all"]["counts"]["positive_hit@5"],
        "useful_hit@5": summary["headline"]["useful"]["counts"]["useful_hit@5"],
        "direct_useful_hit@5": summary["headline"]["direct"]["counts"]["useful_hit@5"],
        "ndcg@10": summary["headline"]["all"]["metrics"]["ndcg@10"]["mean"],
    }


def build_m01_report(args: argparse.Namespace) -> None:
    cases, fixture_manifest = load_fixture(args.output_dir, "test")
    corpus = load_corpus(args.corpus_db)
    corpus_refs = set(corpus)
    score_group = scorer_module(args.scorer_package)
    candidate_report, candidate_rankings = load_run(
        args.candidate, cases, corpus_refs, fixture_manifest
    )
    candidate_scores = score_run(cases, candidate_rankings, score_group)
    parent_report, parent_rankings = load_run(
        args.parent_candidate, cases, corpus_refs, fixture_manifest
    )
    parent_scores = score_run(cases, parent_rankings, score_group)
    selection = json.loads(args.selection.read_text(encoding="utf8"))
    baseline_dir = args.baseline_dir
    baseline_report = json.loads((baseline_dir / "report.json").read_text(encoding="utf8"))
    if baseline_report.get("outcome") != "accepted":
        raise ValueError("M01 parent report is not accepted")
    if baseline_report["dataset"]["fixture_sha256"] != fixture_manifest["fixture_sha256"]:
        raise ValueError("M01 parent/test fixture mismatch")

    baseline_rankings: dict[str, dict[str, list[str]]] = {}
    baseline_scores: dict[str, dict[str, dict[str, float]]] = {}
    baseline_runs: dict[str, dict[str, Any]] = {}
    for name, _ in RUN_COLUMNS:
        baseline_rankings[name], baseline_runs[name] = load_committed_rankings(
            name, baseline_dir, baseline_report, cases, corpus_refs
        )
        baseline_scores[name] = score_run(cases, baseline_rankings[name], score_group)

    candidate_summary = candidate_headline_summary(
        candidate_report, cases, candidate_rankings, candidate_scores, args.bootstrap, args.seed
    )
    summary_runs = {**baseline_report["runs"], "m01": candidate_summary}
    useful = [case for case in cases if any(grade >= 2 for grade in case["qrels"].values())]
    direct = [case for case in cases if case["best_settled_label"] == "direct"]
    comparisons = {}
    for baseline in ("dense", "browser"):
        comparisons[f"m01_vs_{baseline}"] = {
            "positive_hit@5": paired_details(
                cases, baseline_scores[baseline], candidate_scores, "positive_hit@5",
                args.bootstrap, args.seed,
            ),
            "useful_hit@5": paired_details(
                useful, baseline_scores[baseline], candidate_scores, "useful_hit@5",
                args.bootstrap, args.seed,
            ),
            "ndcg@10": paired_details(
                cases, baseline_scores[baseline], candidate_scores, "ndcg@10",
                args.bootstrap, args.seed,
            ),
            "positive_rank_regressions@10": rank_regression_details(
                cases, baseline_rankings[baseline], candidate_rankings, minimum_grade=1
            ),
            "useful_rank_regressions@10": rank_regression_details(
                useful, baseline_rankings[baseline], candidate_rankings, minimum_grade=2
            ),
        }

    parent_summary = candidate_headline_summary(
        parent_report, cases, parent_rankings, parent_scores, args.bootstrap, args.seed
    )
    parent_counts = headline_test_counts(parent_summary)
    candidate_counts = headline_test_counts(candidate_summary)
    test_gate = all(
        candidate_counts[key] >= parent_counts[key]
        for key in ("relevant_hit@5", "useful_hit@5", "direct_useful_hit@5", "ndcg@10")
    )
    budget = baseline_report["runtime_budgets"]["runs"]["dense"]
    runtime_measurements = {
        "p95_query_ms": candidate_summary["timings_ms"]["p95"],
        "index_bytes": candidate_summary["index_bytes"],
        "peak_rss_bytes": candidate_summary["peak_rss_bytes"],
    }
    runtime_gate = all(
        runtime_measurements[key] <= budget[f"{key}_max"]
        for key in runtime_measurements
    )
    selected_config = canonical_config(selection["selected_config"])
    config_match = canonical_config(candidate_report["config"]) == selected_config
    selection_hash_match = candidate_report.get("selection_sha256") == sha256_file(args.selection)
    selection_match = candidate_report.get("selection") == selection
    selection_contract = (
        selection.get("split") == "validation"
        and selection.get("test_accessed") is False
        and selection["dataset"]["fixture_sha256"] == EXPECTED_VALIDATION_FIXTURE_SHA256
        and selection["determinism_check"]["passed"]
    )
    frozen_code_unchanged = all(
        sha256_file(ROOT / relative) == digest
        for relative, digest in selection["code_sha256"].items()
    )
    parent_provenance = baseline_report["runs"]["dense"]["provenance"]
    fixed_contract_keys = (
        "model", "model_revision", "model_files", "dtype", "distance",
        "field", "source_attribution", "corpus",
    )
    fixed_contract_match = all(
        candidate_report.get(key) == parent_provenance.get(key) for key in fixed_contract_keys
    )
    prefix_match = candidate_report.get("query_prefix") == selection["prefixes"]["query"] and candidate_report.get("passage_prefix") == selection["prefixes"]["passage"]
    depth_bound = (
        candidate_report["result_k"] == min(50, candidate_report["config"]["fetch_k"])
        and all(
            len(run["refs"]) <= candidate_report["result_k"]
            for run in candidate_report["runs"].values()
        )
    )
    exact_parent_queries = sum(
        parent_rankings[case["id"]] == baseline_rankings["dense"][case["id"]]
        for case in cases
    )
    parent_config = canonical_config(parent_report["config"])
    expected_parent_config = {"retrieval_mode": "similarity", "fetch_k": 50, "mmr_lambda": 1.0}
    parent_parity = exact_parent_queries == len(cases) and parent_config == expected_parent_config
    validation_gate_passed = not selection["selected_changed"] or selection["validation_gate_passed"]
    checks = json.loads(args.checks.read_text(encoding="utf8"))
    targeted_checks_passed = checks.get("status") == "passed"
    accepted = all((
        config_match,
        selection_hash_match,
        selection_match,
        selection_contract,
        frozen_code_unchanged,
        fixed_contract_match,
        prefix_match,
        depth_bound,
        test_gate,
        runtime_gate,
        parent_parity,
        validation_gate_passed,
        targeted_checks_passed,
    ))
    report = {
        "profile": "pr2_frozen_v1",
        "experiment": "M01",
        "outcome": "accepted" if accepted else "rejected",
        "contribution": "Configurable similarity/MMR retrieval with ayah-ID deduplication; relevance gain is claimed only if the frozen candidate changes and passes every gate.",
        "dataset": fixture_manifest,
        "selection": selection,
        "selection_sha256": sha256_file(args.selection),
        "parent_report_sha256": sha256_file(baseline_dir / "report.json"),
        "scorer_sha256": text_sha256(args.scorer_package / "evaluate_search.py"),
        "code_sha256": {
            "evaluation/islamqa_frozen.py": sha256_file(HERE / "islamqa_frozen.py"),
            "utils.py": sha256_file(ROOT / "utils.py"),
            "evaluation/test_retrieval_config.py": sha256_file(HERE / "test_retrieval_config.py"),
        },
        "runs": summary_runs,
        "comparisons": comparisons,
        "on_off_ablation": {
            "off": "Current-code similarity rollback: FETCH_K=50, MMR_LAMBDA=1.0; exact M00 parity required.",
            "on": selected_config,
            "off_config": parent_config,
            "off_exact_m00_queries": exact_parent_queries,
            "off_headline_counts": parent_counts,
            "off_timings_ms": parent_summary["timings_ms"],
            "comparison": "m01_vs_dense",
        },
        "runtime_budgets": baseline_report["runtime_budgets"],
        "runtime_budget_check": {
            "budget": budget,
            "measurements": runtime_measurements,
            "passed": runtime_gate,
        },
        "acceptance": {
            "accepted": accepted,
            "config_matches_frozen_selection": config_match,
            "selection_hash_matches": selection_hash_match,
            "embedded_selection_matches": selection_match,
            "validation_only_selection_contract": selection_contract,
            "frozen_code_unchanged": frozen_code_unchanged,
            "fixed_model_corpus_contract_matches": fixed_contract_match,
            "prefixes_match": prefix_match,
            "unique_ids_and_depth_bound": depth_bound,
            "validation_gate_passed": validation_gate_passed,
            "test_gate_passed": test_gate,
            "parent_config_matches_rollback": parent_config == expected_parent_config,
            "parent_headline_counts": parent_counts,
            "candidate_headline_counts": candidate_counts,
            "runtime_budget_passed": runtime_gate,
            "targeted_checks_passed": targeted_checks_passed,
            "similarity_parent_exact_queries": exact_parent_queries,
            "similarity_rollback_parity": parent_parity,
        },
        "targeted_checks": checks,
        "source_ranking_artifacts": {
            **{
                name: {
                    "file": f"{name}_query_rankings.jsonl",
                    "sha256": baseline_report["query_ranking_artifacts"][name]["sha256"],
                    "source": "M00 committed query rankings",
                }
                for name, _ in RUN_COLUMNS
            },
            "m01": {"file": args.candidate.name,
                "sha256": sha256_file(args.candidate),
                "bytes": args.candidate.stat().st_size,
            },
            "m01_parent": {
                "file": args.parent_candidate.name,
                "sha256": sha256_file(args.parent_candidate),
                "bytes": args.parent_candidate.stat().st_size,
            },
        },
        "limitations": [
            *fixture_manifest["notes"],
            "The English IslamQA score does not establish Arabic/Bengali semantic quality.",
            "Runtime budgets are machine-specific and remain fixed from M00.",
        ],
    }
    write_jsonl(
        args.output_dir / "m01_query_rankings.jsonl",
        (
            {
                "id": case["id"],
                "group_id": case["group_id"],
                "query": case["query"],
                "ranking": candidate_rankings[case["id"]],
                "source_fields": candidate_report["runs"][case["id"]]["source_fields"],
            }
            for case in cases
        ),
    )
    report["query_ranking_artifacts"] = {
        "m01": {
            "file": "m01_query_rankings.jsonl",
            "sha256": sha256_file(args.output_dir / "m01_query_rankings.jsonl"),
            "bytes": (args.output_dir / "m01_query_rankings.jsonl").stat().st_size,
        }
    }
    table = markdown_leaderboard(report)
    (args.output_dir / "leaderboard.md").write_text(table + "\n", encoding="utf8")
    write_json(args.output_dir / "report.json", report)
    print(table)
    print()
    print(json.dumps(report["acceptance"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Validate v3 and freeze the fixture")
    prepare_parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    prepare_parser.add_argument(
        "--corpus-db", type=Path, default=DEFAULT_AAYATUN_REPO / "db" / "en_quran.db"
    )
    prepare_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    prepare_parser.add_argument("--split", choices=("test", "validation"), default="test")
    prepare_parser.set_defaults(func=prepare)

    lexical_parser = subparsers.add_parser("lexical", help="Run a pinned Aayatun lexical revision")
    lexical_parser.add_argument("--repo", type=Path, default=DEFAULT_AAYATUN_REPO)
    lexical_parser.add_argument("--revision", required=True)
    lexical_parser.add_argument("--field", choices=("translation", "all"), required=True)
    lexical_parser.add_argument("--engine-name", required=True)
    lexical_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    lexical_parser.add_argument("--output", type=Path, required=True)
    lexical_parser.set_defaults(func=run_lexical)

    query_parser = subparsers.add_parser("lexical-queries", help=argparse.SUPPRESS)
    query_parser.add_argument("--runtime", type=Path, required=True)
    query_parser.add_argument("--engine-name", required=True)
    query_parser.add_argument("--field", choices=("translation", "all"), required=True)
    query_parser.set_defaults(func=run_lexical_queries)

    dense_parser = subparsers.add_parser("dense", help="Build and run the offline E5/FAISS baseline")
    dense_parser.add_argument(
        "--corpus-db", type=Path, default=DEFAULT_AAYATUN_REPO / "db" / "en_quran.db"
    )
    dense_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    dense_parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_OUTPUT / "runtime")
    dense_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "dense_rankings.json")
    dense_parser.set_defaults(func=run_dense)

    dense_config_parser = subparsers.add_parser("dense-config", help="Run one frozen dense retrieval configuration")
    dense_config_parser.add_argument(
        "--corpus-db", type=Path, default=DEFAULT_AAYATUN_REPO / "db" / "en_quran.db"
    )
    dense_config_parser.add_argument("--output-dir", type=Path, default=DEFAULT_M01_TEST)
    dense_config_parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_M01_TEST / "runtime")
    dense_config_parser.add_argument("--output", type=Path, default=DEFAULT_M01_TEST / "m01_rankings.json")
    dense_config_parser.add_argument("--split", choices=("test", "validation"), default="test")
    dense_config_parser.add_argument("--retrieval-mode", choices=("similarity", "mmr"), required=True)
    dense_config_parser.add_argument("--fetch-k", type=int, choices=(20, 50, 100), required=True)
    dense_config_parser.add_argument("--mmr-lambda", type=float, choices=(1.0, 0.85, 0.7), required=True)
    dense_config_parser.add_argument("--selection", type=Path)
    dense_config_parser.set_defaults(func=run_dense_config)

    dense_sweep_parser = subparsers.add_parser("dense-sweep", help="Select M01 settings on validation only")
    dense_sweep_parser.add_argument(
        "--corpus-db", type=Path, default=DEFAULT_AAYATUN_REPO / "db" / "en_quran.db"
    )
    dense_sweep_parser.add_argument("--scorer-package", type=Path, default=DEFAULT_SCORER_PACKAGE)
    dense_sweep_parser.add_argument("--output-dir", type=Path, default=DEFAULT_M01_VALIDATION)
    dense_sweep_parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_M01_VALIDATION / "runtime")
    dense_sweep_parser.add_argument("--selection", type=Path, default=DEFAULT_M01_VALIDATION / "selection.json")
    dense_sweep_parser.add_argument("--bootstrap", type=int, default=2000)
    dense_sweep_parser.add_argument("--seed", type=int, default=4213)
    dense_sweep_parser.set_defaults(func=run_dense_sweep)

    browser_parser = subparsers.add_parser("browser", help="Reproduce the pinned PR #2 browser worker")
    browser_parser.add_argument("--aayatun-repo", type=Path, default=DEFAULT_AAYATUN_REPO)
    browser_parser.add_argument("--pr2-eval-repo", type=Path, default=DEFAULT_PR2_EVAL_REPO)
    browser_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    browser_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "browser_rankings.json")
    browser_parser.set_defaults(func=run_browser)

    report_parser = subparsers.add_parser("report", help="Score all runs and write the leaderboard")
    report_parser.add_argument(
        "--corpus-db", type=Path, default=DEFAULT_AAYATUN_REPO / "db" / "en_quran.db"
    )
    report_parser.add_argument("--scorer-package", type=Path, default=DEFAULT_SCORER_PACKAGE)
    report_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    report_parser.add_argument(
        "--browser-reference",
        type=Path,
        default=DEFAULT_PR2_EVAL_REPO / "evaluation" / "islamqa_pr2_meaning_rankings.json",
    )
    report_parser.add_argument("--bootstrap", type=int, default=2000)
    report_parser.add_argument("--seed", type=int, default=4213)
    report_parser.set_defaults(func=build_report)

    m01_report_parser = subparsers.add_parser("m01-report", help="Score the frozen M01 test and append its leaderboard column")
    m01_report_parser.add_argument(
        "--corpus-db", type=Path, default=DEFAULT_AAYATUN_REPO / "db" / "en_quran.db"
    )
    m01_report_parser.add_argument("--scorer-package", type=Path, default=DEFAULT_SCORER_PACKAGE)
    m01_report_parser.add_argument("--output-dir", type=Path, default=DEFAULT_M01_TEST)
    m01_report_parser.add_argument("--candidate", type=Path, default=DEFAULT_M01_TEST / "m01_rankings.json")
    m01_report_parser.add_argument("--parent-candidate", type=Path, default=DEFAULT_M01_TEST / "m01_parent_rankings.json")
    m01_report_parser.add_argument("--selection", type=Path, default=DEFAULT_M01_VALIDATION / "selection.json")
    m01_report_parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_OUTPUT)
    m01_report_parser.add_argument("--checks", type=Path, default=DEFAULT_M01_TEST / "checks.json")
    m01_report_parser.add_argument("--bootstrap", type=int, default=2000)
    m01_report_parser.add_argument("--seed", type=int, default=4213)
    m01_report_parser.set_defaults(func=build_m01_report)

    if __package__:
        from . import m02_ai
    else:
        import m02_ai
    m02_ai.add_parsers(subparsers)
    if __package__:
        from . import m03_tafsir
    else:
        import m03_tafsir
    m03_tafsir.add_parsers(subparsers)
    if __package__:
        from . import m04_translation
    else:
        import m04_translation
    m04_translation.add_parsers(subparsers)
    if __package__:
        from . import m05_spans
    else:
        import m05_spans
    m05_spans.add_parsers(subparsers)

    args = parser.parse_args()
    if hasattr(args, "bootstrap") and args.bootstrap < 0:
        parser.error("--bootstrap must be non-negative")
    args.func(args)


if __name__ == "__main__":
    main()
