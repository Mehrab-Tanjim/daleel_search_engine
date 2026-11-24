"""Load the IslamQA split package into validated, join-friendly structures."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from dataset_common import PACKAGE, SPLITS, read_json, read_jsonl
from validate_dataset import DatasetValidationError, validate_package


@dataclass(frozen=True)
class SplitData:
    """One split, with records indexed by their stable join keys."""

    name: str
    questions: dict[str, dict[str, Any]]
    qrels: dict[str, dict[str, int]]
    candidates: dict[str, tuple[dict[str, Any], ...]]
    judged_pairs: dict[str, frozenset[str]]

    @property
    def group_ids(self) -> tuple[str, ...]:
        return tuple(self.questions)

    @property
    def candidate_count(self) -> int:
        return sum(len(rows) for rows in self.candidates.values())

    @property
    def positive_group_ids(self) -> tuple[str, ...]:
        return tuple(
            group_id
            for group_id, question in self.questions.items()
            if question["label_summary"]["best_settled_label"] != "none"
        )


@dataclass(frozen=True)
class Dataset:
    """Validated package contents with split ownership made explicit."""

    package: Path
    manifest: dict[str, Any]
    splits: dict[str, SplitData]
    conflicts: tuple[dict[str, Any], ...]
    components: dict[str, dict[str, Any]]
    group_splits: dict[str, str]

    @classmethod
    def load(
        cls,
        package: Path = PACKAGE,
        validate: bool = True,
        check_hashes: bool = True,
    ) -> Dataset:
        package = package.resolve()
        if validate:
            validate_package(package, check_hashes=check_hashes)

        manifest = read_json(package / "manifest.json")
        splits: dict[str, SplitData] = {}
        group_splits: dict[str, str] = {}
        for split in SPLITS:
            question_rows = read_jsonl(package / split / "questions.jsonl")
            questions = {row["group_id"]: row for row in question_rows}
            if len(questions) != len(question_rows):
                raise ValueError(f"{split}: duplicate group IDs")

            qrels: dict[str, dict[str, int]] = defaultdict(dict)
            for row in read_jsonl(package / split / "grades.jsonl"):
                group_id = row["group_id"]
                if group_id not in questions:
                    raise ValueError(f"{split}: qrel references unknown group {group_id}")
                if row["ayah"] in qrels[group_id]:
                    raise ValueError(f"{split}: duplicate qrel {group_id}/{row['ayah']}")
                qrels[group_id][row["ayah"]] = row["grade"]

            candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in read_jsonl(package / split / "unjudged_candidates.jsonl"):
                if row["group_id"] not in questions:
                    raise ValueError(
                        f"{split}: candidate references unknown group {row['group_id']}"
                    )
                candidates[row["group_id"]].append(row)

            judged_pairs = {
                group_id: frozenset(
                    verse["ayah"]
                    for verses in question["evidence"].values()
                    for verse in verses
                )
                for group_id, question in questions.items()
            }
            splits[split] = SplitData(
                name=split,
                questions=questions,
                qrels=dict(qrels),
                candidates={
                    group_id: tuple(rows)
                    for group_id, rows in candidates.items()
                },
                judged_pairs=judged_pairs,
            )
            for group_id in questions:
                if group_id in group_splits:
                    raise ValueError(f"Group {group_id} occurs in multiple splits")
                group_splits[group_id] = split

        component_rows = read_jsonl(package / "split_components.jsonl")
        components = {row["component_id"]: row for row in component_rows}
        if len(components) != len(component_rows):
            raise ValueError("Duplicate component IDs")

        return cls(
            package=package,
            manifest=manifest,
            splits=splits,
            conflicts=tuple(
                read_jsonl(package / "conflicts_excluded_from_grades.jsonl")
            ),
            components=components,
            group_splits=group_splits,
        )

    @property
    def questions(self) -> dict[str, dict[str, Any]]:
        return {
            group_id: question
            for split in SPLITS
            for group_id, question in self.splits[split].questions.items()
        }

    def split_for_group(self, group_id: str) -> str:
        try:
            return self.group_splits[group_id]
        except KeyError as exc:
            raise KeyError(f"Unknown group ID: {group_id}") from exc

    def judged_pairs_for(self, group_id: str) -> frozenset[str]:
        return self.splits[self.split_for_group(group_id)].judged_pairs[group_id]

    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": self.manifest["schema_version"],
            "release_status": self.manifest["release_status"],
            "benchmark_grade": self.manifest["benchmark_grade"],
            "splits": {
                split: {
                    "question_groups": len(data.questions),
                    "scored_pairs": sum(len(rows) for rows in data.qrels.values()),
                    "unjudged_candidates": data.candidate_count,
                }
                for split, data in self.splits.items()
            },
            "conflicts": len(self.conflicts),
            "components": len(self.components),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=PACKAGE)
    parser.add_argument("--no-validate", action="store_true")
    parser.add_argument("--skip-hashes", action="store_true")
    args = parser.parse_args()
    try:
        dataset = Dataset.load(
            args.package,
            validate=not args.no_validate,
            check_hashes=not args.skip_hashes,
        )
    except DatasetValidationError as exc:
        print(json.dumps({"valid": False, "errors": exc.errors}, indent=2))
        raise SystemExit(1)
    print(json.dumps(dataset.summary(), indent=2))


if __name__ == "__main__":
    main()
