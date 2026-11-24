"""Export and audit optional verse representations without changing the Quran corpus."""

from __future__ import annotations

from contextlib import closing
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import statistics
from typing import Any, Iterable


AI_DB_SHA256 = "0a334c0e376ccca777312757d8671ad08fc71803effc8fdd8644999b598ff91e"
AI_SOURCE_COMMIT = "23dadd724dd72bed8731ff48c44415934d343381"
AI_EXPORTER_COMMIT = "ace935c95589d03694f83c6c83e1e498c4d3095b"
DATASET_COMMIT = "d936c64c51ebd0908b6ecf0d9b6e4debbb39b517"
URL_RE = re.compile(r"https?://\S+")
WHITESPACE_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"\w+", re.UNICODE)
AI_FIELDS = ("word", "classical_meaning", "meaning_here", "notable_usage")
SOURCE_FIELDS = ("question", "answer")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows)
    path.write_text(payload, encoding="utf8")


def flatten_ai_content(raw: str) -> str:
    """Match Aayatun build_search.load_rows docs.ai flattening exactly."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return ""
    if not isinstance(data, dict):
        return ""
    parts = [data.get("verse_context") or ""]
    for meaning in data.get("word_meanings") or []:
        if not isinstance(meaning, dict):
            continue
        parts.extend(meaning.get(field) or "" for field in AI_FIELDS)
    text = URL_RE.sub(" ", " ".join(str(part) for part in parts))
    return WHITESPACE_RE.sub(" ", text).strip()


def normalized_tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.casefold())


def shingle_digest(tokens: list[str], size: int) -> bytes:
    return hashlib.blake2b("\0".join(tokens).encode("utf8"), digest_size=12).digest()


def audit_source_overlap(
    ai_rows: list[dict[str, Any]],
    source_paths: list[Path],
    shingle_size: int = 12,
) -> dict[str, Any]:
    """Conservatively exclude exact long-span matches; semantic independence is not certified."""
    source_records = {path.name: json.loads(path.read_text(encoding="utf8")) for path in source_paths}
    source_shingles: set[bytes] = set()
    for records in source_records.values():
        for record in records:
            for field in SOURCE_FIELDS:
                text = record.get(field)
                if not isinstance(text, str):
                    continue
                tokens = normalized_tokens(text)
                source_shingles.update(
                    shingle_digest(tokens[index:index + shingle_size], shingle_size)
                    for index in range(max(0, len(tokens) - shingle_size + 1))
                )
    included: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    excluded_shingles: set[bytes] = set()
    for row in ai_rows:
        tokens = normalized_tokens(row["text"])
        shingles = {
            shingle_digest(tokens[index:index + shingle_size], shingle_size)
            for index in range(max(0, len(tokens) - shingle_size + 1))
        }
        matches = shingles & source_shingles
        if matches:
            excluded.append({"ayah": row["ayah"], "matching_shingles": len(matches)})
            excluded_shingles.update(matches)
        else:
            included.append(row)
    overlaps: list[dict[str, Any]] = []
    for source_file, records in source_records.items():
        for index, record in enumerate(records):
            for field in SOURCE_FIELDS:
                text = record.get(field)
                if not isinstance(text, str):
                    continue
                tokens = normalized_tokens(text)
                matches = {
                    shingle_digest(tokens[offset:offset + shingle_size], shingle_size)
                    for offset in range(max(0, len(tokens) - shingle_size + 1))
                } & excluded_shingles
                if matches:
                    overlaps.append({
                        "source_file": source_file,
                        "record_index": index,
                        "field": field,
                        "matching_shingles": len(matches),
                    })
    return {
        "method": f"Exact normalized {shingle_size}-token shingle intersection; no fuzzy or semantic independence claim.",
        "shingle_size": shingle_size,
        "source_records_checked": sum(len(records) for records in source_records.values()),
        "source_shingles": len(source_shingles),
        "overlap_count": len(overlaps),
        "overlaps": overlaps,
        "included_rows": included,
        "excluded_rows": excluded,
        "exclusion_rule": "Exclude an entire AI representation if any exact normalized 12-token span matches an IslamQA question or answer.",
        "checked_source_fields": list(SOURCE_FIELDS),
    }


def export_ai_representation(
    ai_db: Path,
    corpus: dict[str, str],
    output: Path,
    source_paths: list[Path],
    tokenizer: Any | None = None,
    max_tokens: int = 512,
) -> dict[str, Any]:
    if sha256_file(ai_db) != AI_DB_SHA256:
        raise ValueError("AI source database does not match the pinned Aayatun blob")
    rows: list[dict[str, Any]] = []
    malformed: list[str] = []
    empty: list[str] = []
    with closing(sqlite3.connect(f"{ai_db.resolve().as_uri()}?mode=ro", uri=True)) as db:
        source_rows = db.execute(
            "SELECT surah, verse, content FROM verse_analysis ORDER BY surah, verse"
        ).fetchall()
    for surah, zero_based_ayah, raw in source_rows:
        ref = f"{surah}:{zero_based_ayah + 1}"
        if ref not in corpus:
            raise ValueError(f"AI source references unknown ayah: {ref}")
        text = flatten_ai_content(raw)
        if not text:
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                malformed.append(ref)
            else:
                if isinstance(parsed, dict):
                    empty.append(ref)
                else:
                    malformed.append(ref)
            continue
        rows.append({
            "ayah": ref,
            "text": text,
            "field": "ai",
            "source": "aayatun_ai_generated",
            "source_db_sha256": AI_DB_SHA256,
            "source_commit": AI_SOURCE_COMMIT,
        })
    overlap = audit_source_overlap(rows, source_paths)
    rows = overlap.pop("included_rows")
    excluded = overlap.pop("excluded_rows")
    token_counts: list[int] = []
    truncated: list[str] = []
    for row in rows:
        if tokenizer is not None:
            count = len(tokenizer.encode("passage: " + row["text"], add_special_tokens=True, truncation=False))
            token_counts.append(count)
            if count > max_tokens:
                truncated.append(row["ayah"])
    indexed = {row["ayah"] for row in rows}
    missing = sorted(set(corpus) - indexed, key=lambda ref: tuple(map(int, ref.split(":"))))
    write_jsonl(output, rows)
    token_summary = None
    if token_counts:
        ordered = sorted(token_counts)
        truncated_path = output.with_name("truncated_ayahs.json")
        truncated_payload = json.dumps(truncated, ensure_ascii=False, separators=(",", ":")).encode("utf8")
        truncated_path.write_bytes(truncated_payload)
        token_summary = {
            "counter": "AutoTokenizer.encode with passage prefix and special tokens",
            "max_seq_length": max_tokens,
            "documents": len(token_counts),
            "truncated_documents": len(truncated),
            "truncated_ayahs_file": truncated_path.name,
            "truncated_ayahs_sha256": hashlib.sha256(truncated_payload).hexdigest(),
            "truncated_ayahs_bytes": len(truncated_payload),
            "min": min(token_counts),
            "median": statistics.median(token_counts),
            "p95": ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)],
            "max": max(token_counts),
        }
    manifest = {
        "experiment": "M02",
        "representation": "ai",
        "display_field": "translation",
        "source": {
            "dataset": "Aayatun existing db/ai_context.db verse_analysis",
            "source_commit": AI_SOURCE_COMMIT,
            "exporter_commit": AI_EXPORTER_COMMIT,
            "source_db_sha256": AI_DB_SHA256,
            "source_db_bytes": ai_db.stat().st_size,
            "source_rows": len(source_rows),
            "content_type": "AI-generated English explanations and word meanings; not scripture",
            "reuse_status": "Existing Aayatun content; downstream reuse rights remain unresolved.",
        },
        "flattening": {
            "fields": ["verse_context", *AI_FIELDS],
            "ignored_fields": ["verse_context_continuation"],
            "url_removal": True,
            "whitespace_normalization": True,
            "matches_aayatun_docs_ai": True,
        },
        "corpus": {"verses": len(corpus), "sha256": hashlib.sha256(json.dumps(corpus, ensure_ascii=False, sort_keys=True).encode("utf8")).hexdigest()},
        "counts": {
            "source_rows": len(source_rows),
            "valid_json": len(rows) + len(excluded) + len(empty),
            "exported": len(rows),
            "malformed_json": len(malformed),
            "overlap_excluded": len(excluded),
            "missing": len(missing),
            "empty_text": len(empty),
        },
        "malformed_json_ayahs": malformed,
        "empty_text_ayahs": empty,
        "overlap_excluded_rows": excluded,
        "missing_ayahs": missing,
        "characters": sum(len(row["text"]) for row in rows),
        "token_truncation": token_summary,
        "provenance_audit": {
            **overlap,
            "source_hashes": {path.name: sha256_file(path) for path in source_paths},
            "dataset_commit": DATASET_COMMIT,
            "temporal_evidence": "The pinned AI blob predates the v3 dataset merge; this does not establish when its model inputs were created.",
            "limitation": "Exact overlap screening and source history do not certify that no model independently produced similar text.",
        },
        "output": {"file": output.name, "sha256": sha256_file(output), "bytes": output.stat().st_size},
    }
    write_json(output.with_suffix(".manifest.json"), manifest)
    return manifest


def ayah_sort_key(ref: str) -> tuple[int, int]:
    surah, ayah = ref.split(":")
    return int(surah), int(ayah)


def fuse_candidates(
    candidates_by_field: dict[str, list[tuple[Any, float]]],
    k: int,
    field_priority: tuple[str, ...] = ("translation", "ai"),
) -> list[tuple[Any, float]]:
    """Return one maximum-scored document per ayah with deterministic ties."""
    if k <= 0:
        raise ValueError("k must be positive")
    priority = {field: index for index, field in enumerate(field_priority)}
    best: dict[str, tuple[Any, float, int]] = {}
    for field, candidates in candidates_by_field.items():
        if field not in priority:
            raise ValueError(f"Unknown representation field: {field}")
        for doc, score in candidates:
            ref = f"{int(doc.metadata['surah'])}:{int(doc.metadata['ayah'])}"
            rank = priority[field]
            current = best.get(ref)
            if current is None or float(score) > current[1] or (float(score) == current[1] and rank < current[2]):
                best[ref] = (doc, float(score), rank)
    ordered = sorted(
        best.items(),
        key=lambda item: (-item[1][1], item[1][2], ayah_sort_key(item[0])),
    )
    return [(doc, score) for _, (doc, score, _) in ordered[:k]]
