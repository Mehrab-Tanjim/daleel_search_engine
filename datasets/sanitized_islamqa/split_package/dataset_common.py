"""Shared helpers for the IslamQA split package."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sqlite3
import unicodedata
from typing import Any, Iterable

PACKAGE = Path(__file__).resolve().parent
DATA = PACKAGE.parent
SPLITS = ("train", "validation", "test")
TIERS = ("direct", "partial", "background", "unrelated")
GRADES = {"direct": 3, "partial": 2, "background": 1, "unrelated": 0}
SCHEMA_VERSION = 2
AYAH_RE = re.compile(r"([1-9]\d*):([1-9]\d*)\Z")
ARABIC_RE = re.compile(r"[\u0600-\u06ff]")
LATIN_RE = re.compile(r"[A-Za-z]")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf8").splitlines()
        if line.strip()
    ]


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf8")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    path.write_bytes(text.encode("utf8"))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(
        json.dumps(row, ensure_ascii=False) + "\n"
        for row in rows
    )
    path.write_bytes(text.encode("utf8"))


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.findall(r"\w+", normalized))


def detect_language(text: str) -> str:
    arabic = bool(ARABIC_RE.search(text))
    latin = bool(LATIN_RE.search(text))
    if arabic and latin:
        return "mixed"
    if arabic:
        return "ar"
    if latin:
        return "en"
    return "und"


def ayah_parts(ayah: str) -> tuple[int, int]:
    match = AYAH_RE.fullmatch(ayah)
    if not match:
        raise ValueError(f"Invalid ayah reference: {ayah!r}")
    surah, verse = map(int, match.groups())
    if not 1 <= surah <= 114:
        raise ValueError(f"Invalid surah in ayah reference: {ayah!r}")
    return surah, verse


def ayah_sort_key(ayah: str) -> tuple[int, int]:
    return ayah_parts(ayah)


def load_corpus(path: Path) -> dict[str, str]:
    with sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True) as connection:
        rows = connection.execute("select sura, ayah, text from verses").fetchall()
    corpus = {f"{surah}:{ayah}": text for surah, ayah, text in rows}
    if len(corpus) != len(rows):
        raise ValueError("Translation corpus contains duplicate ayah references")
    return corpus


def origin_profile(origin_splits: Iterable[str]) -> str:
    values = sorted(set(origin_splits))
    if len(values) == 1:
        return values[0]
    return "mixed"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def semantic_sha256(path: Path) -> str:
    """Hash parsed JSON content independently of whitespace and line endings."""
    if path.suffix == ".jsonl":
        value = [canonical_json_bytes(row) for row in read_jsonl(path)]
        payload = b"\n".join(value) + b"\n"
    else:
        payload = canonical_json_bytes(read_json(path))
    return hashlib.sha256(payload).hexdigest()


def stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\n".join(parts).encode("utf8")).hexdigest()
    return f"{prefix}{digest[:16]}"
