"""Build a deduplicated human-review queue from top-k search rankings."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from dataset_common import (
    PACKAGE,
    SPLITS,
    ayah_sort_key,
    load_corpus,
    write_jsonl,
)
from evaluate_search import load_ranking, parse_run_spec
from load_dataset import Dataset


def build_review_pool(
    dataset: Dataset,
    runs: dict[str, dict[str, list[str]]],
    split_names: list[str],
    top_k: int,
    corpus: dict[str, str],
) -> list[dict[str, Any]]:
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    selected = set(split_names)
    pool: dict[tuple[str, str], dict[str, Any]] = {}
    for split in SPLITS:
        if split not in selected:
            continue
        for group_id in dataset.splits[split].group_ids:
            question = dataset.splits[split].questions[group_id]
            judged = dataset.judged_pairs_for(group_id)
            queries = [
                {"query_id": query["query_id"], "text": query["text"]}
                for query in question["queries"]
            ]
            for run_name, rankings in runs.items():
                for rank, ayah in enumerate(rankings.get(group_id, [])[:top_k], start=1):
                    if ayah in judged:
                        continue
                    if ayah not in corpus:
                        raise ValueError(
                            f"{run_name}/{group_id}: ayah {ayah} is absent from the corpus"
                        )
                    key = (group_id, ayah)
                    if key not in pool:
                        pool[key] = {
                            "split": split,
                            "group_id": group_id,
                            "queries": queries,
                            "ayah": ayah,
                            "text": corpus[ayah],
                            "sources": [],
                        }
                    pool[key]["sources"].append({"run": run_name, "rank": rank})

    return sorted(
        pool.values(),
        key=lambda row: (
            SPLITS.index(row["split"]),
            row["group_id"],
            ayah_sort_key(row["ayah"]),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=PACKAGE)
    parser.add_argument("--skip-hashes", action="store_true")
    parser.add_argument(
        "--split",
        action="append",
        choices=SPLITS,
        dest="splits",
        help="Split to review; repeat to combine splits. Default: validation.",
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Ranking JSONL file; NAME defaults to the file stem.",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--corpus", required=True, type=Path, help="Aayatun en_quran.db")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    dataset = Dataset.load(args.package, check_hashes=not args.skip_hashes)
    valid_group_ids = set(dataset.questions)
    runs: dict[str, dict[str, list[str]]] = {}
    for spec in args.run:
        name, path = parse_run_spec(spec)
        if name in runs:
            parser.error(f"Duplicate run name: {name}")
        runs[name] = load_ranking(path, valid_group_ids)

    rows = build_review_pool(
        dataset,
        runs,
        args.splits or ["validation"],
        args.top_k,
        load_corpus(args.corpus),
    )
    write_jsonl(args.output, rows)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "runs": list(runs),
                "top_k": args.top_k,
                "pending_pairs": len(rows),
                "by_split": dict(Counter(row["split"] for row in rows)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
