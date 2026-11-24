"""Evaluate search ranking JSONL files against the settled IslamQA qrels."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any, Iterable

from dataset_common import (
    PACKAGE,
    SPLITS,
    ayah_parts,
    read_jsonl,
    write_json,
)
from load_dataset import Dataset

METRICS = (
    "direct_hit@5",
    "useful_hit@5",
    "positive_hit@5",
    "recall@5",
    "recall@10",
    "mrr@10",
    "ndcg@10",
)


def dcg(grades: Iterable[int], cutoff: int) -> float:
    return sum(
        (2**grade - 1) / math.log2(rank + 1)
        for rank, grade in enumerate(grades, start=1)
        if rank <= cutoff
    )


def score_group(ranking: list[str], qrels: dict[str, int]) -> dict[str, float]:
    """Score one group; unjudged results occupy rank positions but earn no gain."""
    grades_at_5 = [qrels.get(ayah) for ayah in ranking[:5]]
    grades_at_10 = [qrels.get(ayah) for ayah in ranking[:10]]
    positive_total = sum(grade > 0 for grade in qrels.values())
    if positive_total == 0:
        raise ValueError("Group has no settled positive qrel")

    positive_5 = sum(grade is not None and grade > 0 for grade in grades_at_5)
    positive_10 = sum(grade is not None and grade > 0 for grade in grades_at_10)
    first_positive = next(
        (
            rank
            for rank, grade in enumerate(grades_at_10, start=1)
            if grade is not None and grade > 0
        ),
        None,
    )
    ideal = dcg(sorted(qrels.values(), reverse=True), 10)
    actual = dcg(
        [grade if grade is not None else 0 for grade in grades_at_10],
        10,
    )
    return {
        "direct_hit@5": float(any(grade == 3 for grade in grades_at_5)),
        "useful_hit@5": float(
            any(grade is not None and grade >= 2 for grade in grades_at_5)
        ),
        "positive_hit@5": float(positive_5 > 0),
        "recall@5": positive_5 / positive_total,
        "recall@10": positive_10 / positive_total,
        "mrr@10": 0.0 if first_positive is None else 1.0 / first_positive,
        "ndcg@10": 0.0 if ideal == 0 else actual / ideal,
    }


def load_ranking(path: Path, valid_group_ids: set[str]) -> dict[str, list[str]]:
    rankings: dict[str, list[str]] = {}
    for line_number, row in enumerate(read_jsonl(path), start=1):
        if set(row) != {"group_id", "ranking"}:
            raise ValueError(f"{path}:{line_number}: expected group_id and ranking only")
        group_id = row["group_id"]
        if group_id not in valid_group_ids:
            raise ValueError(f"{path}:{line_number}: unknown group ID {group_id}")
        if group_id in rankings:
            raise ValueError(f"{path}:{line_number}: duplicate group ID {group_id}")
        ranking = row["ranking"]
        if not isinstance(ranking, list):
            raise ValueError(f"{path}:{line_number}: ranking must be an array")
        if len(set(ranking)) != len(ranking):
            raise ValueError(f"{path}:{line_number}: duplicate ayah in ranking")
        for ayah in ranking:
            try:
                ayah_parts(ayah)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
        rankings[group_id] = ranking
    return rankings


def parse_run_spec(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        name, value = spec.split("=", 1)
        if not name:
            raise ValueError(f"Invalid run specification: {spec!r}")
        return name, Path(value)
    path = Path(spec)
    return path.stem, path


def percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("Cannot calculate a percentile of an empty sample")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def macro_metrics(
    group_scores: dict[str, dict[str, float]],
) -> dict[str, float]:
    if not group_scores:
        raise ValueError("No groups to evaluate")
    return {
        metric: statistics.fmean(scores[metric] for scores in group_scores.values())
        for metric in METRICS
    }


def bootstrap_intervals(
    group_scores: dict[str, dict[str, float]],
    samples: int,
    seed: int,
) -> dict[str, list[float]]:
    if samples == 0:
        return {}
    group_ids = list(group_scores)
    rng = random.Random(seed)
    draws: dict[str, list[float]] = {metric: [] for metric in METRICS}
    for _ in range(samples):
        sampled = [group_scores[rng.choice(group_ids)] for _ in group_ids]
        means = macro_metrics(dict(enumerate(sampled)))
        for metric in METRICS:
            draws[metric].append(means[metric])
    return {
        metric: [
            round(percentile(values, 0.025), 6),
            round(percentile(values, 0.975), 6),
        ]
        for metric, values in draws.items()
    }


def evaluate_run(
    dataset: Dataset,
    split_names: list[str],
    rankings: dict[str, list[str]],
    bootstrap_samples: int = 2000,
    seed: int = 4213,
) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    selected_groups = {
        group_id
        for split in split_names
        for group_id in dataset.splits[split].questions
    }
    eligible_groups = {
        group_id
        for split in split_names
        for group_id in dataset.splits[split].positive_group_ids
    }
    group_scores = {
        group_id: score_group(
            rankings.get(group_id, []),
            dataset.splits[dataset.split_for_group(group_id)].qrels[group_id],
        )
        for group_id in sorted(eligible_groups)
    }
    means = macro_metrics(group_scores)
    intervals = bootstrap_intervals(group_scores, bootstrap_samples, seed)

    strata: dict[str, list[dict[str, float]]] = defaultdict(list)
    for group_id, scores in group_scores.items():
        question = dataset.questions[group_id]
        strata[question["label_summary"]["best_settled_label"]].append(scores)

    result_counts: Counter[str] = Counter()
    for group_id in selected_groups:
        split = dataset.split_for_group(group_id)
        qrels = dataset.splits[split].qrels[group_id]
        judged = dataset.judged_pairs_for(group_id)
        for ayah in rankings.get(group_id, []):
            result_counts["total_results"] += 1
            if ayah in qrels:
                result_counts["settled_judged_results"] += 1
            elif ayah in judged:
                result_counts["disputed_results"] += 1
            else:
                result_counts["unjudged_results"] += 1

    ranked_selected = selected_groups & set(rankings)
    report = {
        "splits": split_names,
        "eligible_groups": len(eligible_groups),
        "ranked_eligible_groups": len(eligible_groups & set(rankings)),
        "missing_eligible_groups": len(eligible_groups - set(rankings)),
        "excluded_groups_without_settled_positive": len(selected_groups - eligible_groups),
        "metrics": {
            metric: {
                "mean": round(means[metric], 6),
                "ci95": intervals.get(metric),
            }
            for metric in METRICS
        },
        "by_stratum": {
            stratum: {
                "groups": len(scores),
                "metrics": {
                    metric: round(value, 6)
                    for metric, value in macro_metrics(
                        {str(index): row for index, row in enumerate(scores)}
                    ).items()
                },
            }
            for stratum, scores in sorted(strata.items())
        },
        "result_coverage": {
            "ranked_selected_groups": len(ranked_selected),
            "selected_groups": len(selected_groups),
            **dict(result_counts),
        },
        "bootstrap": {
            "samples": bootstrap_samples,
            "seed": seed,
            "method": "percentile_group_resampling",
        },
    }
    return report, group_scores


def paired_comparison(
    baseline_name: str,
    candidate_name: str,
    baseline_scores: dict[str, dict[str, float]],
    candidate_scores: dict[str, dict[str, float]],
    metric: str,
    bootstrap_samples: int = 2000,
    seed: int = 4213,
) -> dict[str, Any]:
    if metric not in METRICS:
        raise ValueError(f"Unknown metric: {metric}")
    common = sorted(set(baseline_scores) & set(candidate_scores))
    if not common:
        raise ValueError("Runs have no common eligible groups")
    deltas = [
        candidate_scores[group_id][metric] - baseline_scores[group_id][metric]
        for group_id in common
    ]
    tolerance = 1e-12
    interval = None
    if bootstrap_samples:
        rng = random.Random(seed)
        means = [
            statistics.fmean(rng.choice(deltas) for _ in common)
            for _ in range(bootstrap_samples)
        ]
        interval = [
            round(percentile(means, 0.025), 6),
            round(percentile(means, 0.975), 6),
        ]
    return {
        "baseline": baseline_name,
        "candidate": candidate_name,
        "metric": metric,
        "common_groups": len(common),
        "wins": sum(delta > tolerance for delta in deltas),
        "losses": sum(delta < -tolerance for delta in deltas),
        "ties": sum(abs(delta) <= tolerance for delta in deltas),
        "baseline_mean": round(
            statistics.fmean(baseline_scores[group_id][metric] for group_id in common),
            6,
        ),
        "candidate_mean": round(
            statistics.fmean(candidate_scores[group_id][metric] for group_id in common),
            6,
        ),
        "delta_mean": round(statistics.fmean(deltas), 6),
        "delta_ci95": interval,
        "bootstrap": {
            "samples": bootstrap_samples,
            "seed": seed,
            "method": "paired_percentile_group_resampling",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=PACKAGE)
    parser.add_argument("--skip-hashes", action="store_true")
    parser.add_argument(
        "--split",
        action="append",
        choices=SPLITS,
        dest="splits",
        help="Split to evaluate; repeat to combine splits. Default: validation.",
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Ranking JSONL file; NAME defaults to the file stem.",
    )
    parser.add_argument("--compare", nargs=2, metavar=("BASELINE", "CANDIDATE"))
    parser.add_argument("--paired-metric", choices=METRICS, default="ndcg@10")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=4213)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.bootstrap < 0:
        parser.error("--bootstrap must be non-negative")

    dataset = Dataset.load(args.package, check_hashes=not args.skip_hashes)
    split_names = args.splits or ["validation"]
    valid_group_ids = set(dataset.questions)
    reports: dict[str, dict[str, Any]] = {}
    scores_by_run: dict[str, dict[str, dict[str, float]]] = {}
    for spec in args.run:
        name, path = parse_run_spec(spec)
        if name in reports:
            parser.error(f"Duplicate run name: {name}")
        rankings = load_ranking(path, valid_group_ids)
        report, scores = evaluate_run(
            dataset,
            split_names,
            rankings,
            bootstrap_samples=args.bootstrap,
            seed=args.seed,
        )
        reports[name] = {"run": name, **report}
        scores_by_run[name] = scores

    result: dict[str, Any] = {"runs": reports}
    if args.compare:
        baseline, candidate = args.compare
        missing = {baseline, candidate} - set(reports)
        if missing:
            parser.error(f"Comparison runs were not supplied: {sorted(missing)}")
        result["paired_comparison"] = paired_comparison(
            baseline,
            candidate,
            scores_by_run[baseline],
            scores_by_run[candidate],
            args.paired_metric,
            bootstrap_samples=args.bootstrap,
            seed=args.seed,
        )

    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
