"""Prepare Quran pairs for review, or assemble reviewed pairs without leaking test questions."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import unicodedata

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "sanitized_islamqa"


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines()]


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf8")


def question_key(question):
    text = unicodedata.normalize("NFKC", question).casefold()
    return " ".join(re.findall(r"\w+", text))


def load_pairs(database):
    with sqlite3.connect(database) as conn:
        corpus = {f"{s}:{v}": text for s, v, text in conn.execute("select surah,verse,en from docs")}
    pairs = []
    for split in ("train", "benchmark"):
        source = HERE / f"islamqa_references_{split}.json"
        for row_index, row in enumerate(json.loads(source.read_text(encoding="utf8"))):
            for reference_index, citation in enumerate(row.get("extracted_references", [])):
                if citation.get("type") != "Quran":
                    continue
                raw = citation.get("reference") or ""
                match = re.search(r"(?<!\d)(\d{1,3})\s*:\s*(\d{1,3})(?:\s*[-–]\s*(\d{1,3}))?", raw)
                refs = []
                if match:
                    surah, first = int(match[1]), int(match[2])
                    last = int(match[3] or first)
                    if first <= last and last - first < 286:
                        refs = [f"{surah}:{verse}" for verse in range(first, last + 1)]
                    if not all(ref in corpus for ref in refs):
                        refs = []
                pairs.append({
                    "id": f"{split}:{row_index}:{reference_index}", "split": split,
                    "source_row": row_index, "reference_index": reference_index,
                    "question": " ".join((row.get("question") or "").split()),
                    "source_reference": raw, "quoted_text": citation.get("text") or "",
                    "ayahs": refs, "verse_texts": [corpus[ref] for ref in refs],
                })
    return pairs, corpus


def assemble(pairs, corpus):
    decisions = read_jsonl(OUTPUT / "review_decisions.jsonl")
    indexed = {row["id"]: row for row in decisions}
    assert len(indexed) == len(decisions), "Duplicate review decisions"
    assert set(indexed) == {row["id"] for row in pairs}, "Review coverage is incomplete"
    benchmark_keys = {question_key(row["question"]) for row in pairs if row["split"] == "benchmark"}
    accepted = {"train": [], "validation": [], "benchmark": []}
    seen = set()
    audit = []
    for pair in pairs:
        decision = indexed[pair["id"]]
        assert decision["relevance"] in {"direct", "partial", "background", "unrelated", "unresolved"}
        assert decision["reason"].strip(), pair["id"]
        refs = decision.get("corrected_ayahs", pair["ayahs"])
        assert all(ref in corpus for ref in refs), pair["id"]
        record = {**pair, "ayahs": refs, "verse_texts": [corpus[ref] for ref in refs],
                  "review": decision, "eligible": False}
        key = question_key(pair["question"])
        identity = (pair["split"], key, tuple(refs))
        if decision["relevance"] != "direct":
            record["exclusion_reason"] = "The review did not judge this a direct answer."
        elif not refs or not key:
            record["exclusion_reason"] = "Missing question or verified verse reference."
        elif pair["split"] == "train" and key in benchmark_keys:
            record["exclusion_reason"] = "Question also occurs in the benchmark."
        elif identity in seen:
            record["exclusion_reason"] = "Duplicate question and verse references within this split."
        else:
            record["eligible"] = True
            destination = pair["split"]
            if destination == "train":
                digest = hashlib.sha256(key.encode("utf8")).digest()
                if int.from_bytes(digest[:4], "big") % 100 < 15:
                    destination = "validation"
            record["dataset_split"] = destination
            accepted[destination].append(record)
            seen.add(identity)
        audit.append(record)
    write_jsonl(OUTPUT / "all_pairs_reviewed.jsonl", audit)
    for split, records in accepted.items():
        write_jsonl(OUTPUT / f"{split}_direct.jsonl", records)
    summary = {
        "source_sha256": {split: hashlib.sha256((HERE / f"islamqa_references_{split}.json").read_bytes()).hexdigest()
                          for split in ("train", "benchmark")},
        "total_quran_pairs": len(pairs), "reviewed_pairs": len(decisions),
        "decisions": dict(Counter(row["relevance"] for row in decisions)),
        "accepted": {split: len(records) for split, records in accepted.items()},
        "exclusions": dict(Counter(row["exclusion_reason"] for row in audit if not row["eligible"])),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf8")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "assemble"))
    parser.add_argument("--corpus", required=True, type=Path, help="Aayatun SQLite search index with docs(surah,verse,en)")
    args = parser.parse_args()
    OUTPUT.mkdir(exist_ok=True)
    pairs, corpus = load_pairs(args.corpus)
    if args.command == "prepare":
        for shard in range(3):
            write_jsonl(OUTPUT / f"review_input_{shard}.jsonl", pairs[shard::3])
    else:
        assemble(pairs, corpus)


if __name__ == "__main__":
    main()
