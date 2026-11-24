"""Deterministic, source-grounded sentence spans for Quran documents."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Iterable


SEGMENTER_VERSION = "deterministic-sentence-spans-v1"
SPAN_MODES = ("whole", "sentences")
SPAN_SENTENCE_COUNTS = (1, 2)
DEFAULT_PASSAGE_PREFIX = "passage: "
TERMINAL_PUNCTUATION = frozenset(".?!。！？؟۔")
CLOSING_QUOTES = frozenset('"”’')
OPENING_QUOTES = frozenset('"“‘')
CLOSING_BRACKETS = frozenset(")]}")
CLOSERS = CLOSING_QUOTES | CLOSING_BRACKETS
OPENING_BRACKETS = frozenset("([{")
ABBREVIATIONS = frozenset({
    "mr", "mrs", "ms", "dr", "prof", "rev", "st", "vs", "no", "fig",
    "vol", "cf", "al", "etc", "e.g", "i.e",
})
WHITESPACE_RE = re.compile(r"\s")
SPAN_ID_RE = re.compile(r"(?P<field>[^:]+):(?P<surah>[1-9]\d*):(?P<ayah>[1-9]\d*):(?P<suffix>whole|s(?P<chunk>\d{4}))\Z")


def canonical_span_config(mode: Any, sentences: Any) -> tuple[str, int]:
    if mode not in SPAN_MODES:
        raise ValueError("span_mode must be 'whole' or 'sentences'")
    if isinstance(sentences, bool):
        raise ValueError("span_sentences must be 1 or 2")
    try:
        count = int(sentences)
    except (TypeError, ValueError) as exc:
        raise ValueError("span_sentences must be 1 or 2") from exc
    if isinstance(sentences, float) and sentences != count:
        raise ValueError("span_sentences must be 1 or 2")
    if count not in SPAN_SENTENCE_COUNTS:
        raise ValueError("span_sentences must be 1 or 2")
    return mode, count


def source_text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf8")).hexdigest()


def quran_index_directory(mode: Any, sentences: Any) -> str:
    mode, sentences = canonical_span_config(mode, sentences)
    return "quran" if mode == "whole" else f"quran-sentences-{sentences}"


def _trim_range(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _is_abbreviation(text: str, period: int) -> bool:
    start = period - 1
    while start >= 0 and (text[start].isalpha() or text[start] == "."):
        start -= 1
    word = text[start + 1:period].lower()
    if word in ABBREVIATIONS:
        return True
    return len(word) == 1 and word.isalpha()


def _is_decimal(text: str, period: int) -> bool:
    return (
        period > 0
        and period + 1 < len(text)
        and text[period - 1].isdigit()
        and text[period + 1].isdigit()
    )


def sentence_ranges(text: str) -> list[tuple[int, int]]:
    """Return conservative sentence ranges without rewriting the source text."""
    if not isinstance(text, str):
        raise ValueError("sentence source must be a string")
    if not text.strip():
        return []

    ranges: list[tuple[int, int]] = []
    start = 0
    parentheses = 0
    curly_quote = False
    straight_quote = False
    single_quote = False
    index = 0
    while index < len(text):
        char = text[index]
        if char in OPENING_BRACKETS:
            parentheses += 1
        elif char in CLOSING_BRACKETS and parentheses:
            parentheses -= 1
        elif char == "“":
            curly_quote = True
        elif char == "”":
            curly_quote = False
        elif char == '"':
            straight_quote = not straight_quote
        elif char in ("'", "‘"):
            previous = text[index - 1] if index else ""
            if not previous.isalnum():
                single_quote = not single_quote
        elif char == "’" and single_quote:
            single_quote = False

        boundary = False
        if char in TERMINAL_PUNCTUATION and not (parentheses or curly_quote or straight_quote or single_quote):
            if char == "." and (_is_abbreviation(text, index) or _is_decimal(text, index)):
                boundary = False
            else:
                following = index + 1
                while following < len(text) and text[following] in CLOSERS:
                    following += 1
                boundary = following == len(text) or text[following].isspace()
        elif char in CLOSERS:
            previous = index - 1
            while previous >= 0 and text[previous].isspace():
                previous -= 1
            boundary = (
                previous >= 0
                and text[previous] in TERMINAL_PUNCTUATION
                and not (parentheses or curly_quote or straight_quote or single_quote)
            )

        if boundary:
            end = index + 1
            while end < len(text) and text[end] in CLOSERS:
                end += 1
            trimmed_start, trimmed_end = _trim_range(text, start, end)
            if trimmed_start < trimmed_end:
                ranges.append((trimmed_start, trimmed_end))
            start = end
            while start < len(text) and text[start].isspace():
                start += 1
            index = start
            continue
        index += 1

    trimmed_start, trimmed_end = _trim_range(text, start, len(text))
    if trimmed_start < trimmed_end:
        ranges.append((trimmed_start, trimmed_end))
    return ranges


def group_sentence_ranges(text: str, sentences: int) -> list[tuple[int, int, int, int]]:
    _, sentences = canonical_span_config("sentences", sentences)
    ranges = sentence_ranges(text)
    return [
        (ranges[index][0], ranges[min(index + sentences, len(ranges)) - 1][1], index + 1, min(index + sentences, len(ranges)))
        for index in range(0, len(ranges), sentences)
    ]


def _input_ids(encoded: Any) -> list[Any]:
    if isinstance(encoded, dict):
        return list(encoded.get("input_ids") or [])
    return list(getattr(encoded, "input_ids", encoded) or [])


def token_count(tokenizer: Any, text: str, prefix: str = DEFAULT_PASSAGE_PREFIX) -> int:
    if tokenizer is None:
        raise ValueError("A tokenizer is required to enforce span token bounds")
    if hasattr(tokenizer, "encode"):
        return len(tokenizer.encode(prefix + text, add_special_tokens=True, truncation=False))
    encoded = tokenizer(prefix + text, add_special_tokens=True, truncation=False)
    return len(_input_ids(encoded))


def _offset_mapping(tokenizer: Any, text: str, prefix: str) -> list[tuple[int, int]]:
    encoded = tokenizer(
        prefix + text,
        add_special_tokens=True,
        return_offsets_mapping=True,
        truncation=False,
    )
    offsets = encoded.get("offset_mapping") if isinstance(encoded, dict) else getattr(encoded, "offset_mapping", None)
    if offsets is None:
        raise ValueError("Tokenizer must provide offset mappings for bounded source spans")
    return [(int(start), int(end)) for start, end in offsets]


def bounded_ranges(
    text: str,
    start: int,
    end: int,
    tokenizer: Any,
    max_tokens: int,
    prefix: str = DEFAULT_PASSAGE_PREFIX,
) -> list[tuple[int, int, int]]:
    """Split one sentence group into exact-source pieces within the token budget."""
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 1:
        raise ValueError("max_tokens must be an integer greater than one")
    if token_count(tokenizer, text[start:end], prefix) <= max_tokens:
        return [(start, end, token_count(tokenizer, text[start:end], prefix))]

    ranges: list[tuple[int, int, int]] = []
    cursor = start
    fixed_tokens = token_count(tokenizer, "", prefix)
    capacity = max_tokens - fixed_tokens
    if capacity <= 0:
        raise ValueError("The passage prefix leaves no capacity for source tokens")
    while cursor < end:
        remaining = text[cursor:end]
        offsets = _offset_mapping(tokenizer, remaining, prefix)
        local_offsets = []
        for token_start, token_end in offsets:
            if token_end <= len(prefix) or token_end <= token_start:
                continue
            local_start = max(0, token_start - len(prefix))
            local_end = token_end - len(prefix)
            if local_end > local_start:
                local_offsets.append((local_start, local_end))
        if not local_offsets:
            raise ValueError("Tokenizer produced no source offsets")
        cut_index = min(capacity, len(local_offsets))
        if cut_index <= 0:
            raise ValueError("No source token fits within the span token budget")
        cut = local_offsets[cut_index - 1][1]
        while cut <= 0 and cut_index > 1:
            cut_index -= 1
            cut = local_offsets[cut_index - 1][1]
        if cut <= 0:
            raise ValueError("Tokenizer produced an invalid source offset")
        candidate_end = cursor + cut
        while token_count(tokenizer, text[cursor:candidate_end], prefix) > max_tokens and cut_index > 1:
            cut_index -= 1
            candidate_end = cursor + local_offsets[cut_index - 1][1]
        count = token_count(tokenizer, text[cursor:candidate_end], prefix)
        if count > max_tokens:
            raise ValueError("A single source token exceeds the span token budget")
        trimmed_start, trimmed_end = _trim_range(text, cursor, candidate_end)
        if trimmed_start >= trimmed_end:
            cursor = candidate_end
            continue
        ranges.append((trimmed_start, trimmed_end, count))
        cursor = candidate_end
        while cursor < end and text[cursor].isspace():
            cursor += 1
    return ranges


def _ayah_parts(metadata: dict[str, Any]) -> tuple[int, int]:
    try:
        surah = int(metadata.get("surah", metadata.get("SurahNo")))
        ayah = int(metadata.get("ayah", metadata.get("AyahNo")))
    except (TypeError, ValueError) as exc:
        raise ValueError("Sentence spans require a valid parent ayah") from exc
    if surah <= 0 or ayah <= 0:
        raise ValueError("Sentence spans require a valid parent ayah")
    return surah, ayah


def build_span_rows(
    text: str,
    metadata: dict[str, Any],
    field: str,
    source: str,
    sentences: int,
    tokenizer: Any,
    max_tokens: int = 512,
    prefix: str = DEFAULT_PASSAGE_PREFIX,
) -> list[dict[str, Any]]:
    """Retain one whole row and add stable, grounded sentence-span rows."""
    _, sentences = canonical_span_config("sentences", sentences)
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Sentence spans cannot be created from empty source text")
    if not isinstance(field, str) or not field or not isinstance(source, str) or not source:
        raise ValueError("Sentence spans require source field attribution")
    surah, ayah = _ayah_parts(metadata)
    digest = source_text_sha256(text)
    parent_id = f"{field}:{surah}:{ayah}:whole"
    rows = [{
        "id": parent_id,
        "ayah": f"{surah}:{ayah}",
        "text": text,
        "field": field,
        "source": source,
        "metadata": dict(metadata),
        "span_kind": "whole",
        "span_start": 0,
        "span_end": len(text),
        "chunk": 0,
        "sentence_start": None,
        "sentence_end": None,
        "sentence_count": len(sentence_ranges(text)),
        "span_sentences": sentences,
        "source_text_sha256": digest,
        "token_count": token_count(tokenizer, text, prefix),
    }]
    chunk = 0
    for start, end, sentence_start, sentence_end in group_sentence_ranges(text, sentences):
        for piece_start, piece_end, count in bounded_ranges(text, start, end, tokenizer, max_tokens, prefix):
            chunk += 1
            span_text = text[piece_start:piece_end]
            if not span_text.strip():
                raise ValueError("Sentence span unexpectedly contains no source text")
            rows.append({
                "id": f"{field}:{surah}:{ayah}:s{chunk:04d}",
                "ayah": f"{surah}:{ayah}",
                "text": span_text,
                "field": field,
                "source": source,
                "metadata": dict(metadata),
                "span_kind": "sentence",
                "span_start": piece_start,
                "span_end": piece_end,
                "chunk": chunk,
                "sentence_start": sentence_start,
                "sentence_end": sentence_end,
                "sentence_count": sentence_end - sentence_start + 1,
                "span_sentences": sentences,
                "source_text_sha256": digest,
                "token_count": count,
            })
    if chunk == 0:
        raise ValueError("Sentence spans unexpectedly contain no sentence rows")
    return rows


def expand_documents(
    documents: Iterable[Any],
    sentences: int,
    tokenizer: Any,
    field: str = "translation",
    source: str = "saheeh_international",
    max_tokens: int = 512,
    prefix: str = DEFAULT_PASSAGE_PREFIX,
) -> list[Any]:
    """Expand LangChain documents while retaining each whole document first."""
    from langchain_core.documents import Document

    expanded = []
    for document in documents:
        metadata = dict(document.metadata)
        surah, ayah = _ayah_parts(metadata)
        metadata["surah"] = surah
        metadata["ayah"] = ayah
        metadata["field"] = field
        metadata["source"] = source
        original = metadata.get("original_text")
        source_text = original if isinstance(original, str) and original.strip() else document.page_content
        for row in build_span_rows(source_text, metadata, field, source, sentences, tokenizer, max_tokens, prefix):
            row_metadata = {
                key: value for key, value in metadata.items()
                if key not in ("original_text", "summary_text", "sentence_ranges", "span_sentences")
            }
            row_metadata.update({key: value for key, value in row.items() if key not in ("metadata", "text", "ayah")})
            expanded.append(Document(page_content=prefix + row["text"], metadata=row_metadata, id=row["id"]))
    return expanded


def audit_span_rows(rows: list[dict[str, Any]], source_texts: dict[str, str], max_tokens: int = 512) -> dict[str, Any]:
    """Verify grounding, bounds, stable IDs, whole retention, and unique parents."""
    ids = [row.get("id") for row in rows]
    if any(not isinstance(doc_id, str) or not doc_id for doc_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("Span IDs are empty, malformed, or duplicated")
    by_parent: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        ref = row.get("ayah")
        text = row.get("text")
        source_text = source_texts.get(ref)
        if ref not in source_texts or not isinstance(text, str) or not text.strip():
            raise ValueError("Span has an unknown parent or empty text")
        if not isinstance(row.get("field"), str) or not row["field"] or not isinstance(row.get("source"), str) or not row["source"]:
            raise ValueError(f"Span source attribution is missing: {ref}")
        start, end = row.get("span_start"), row.get("span_end")
        if isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end <= len(source_text):
            raise ValueError(f"Invalid span offsets: {ref}")
        if source_text[start:end] != text:
            raise ValueError(f"Span is not an exact source substring: {ref}")
        if row.get("source_text_sha256") != source_text_sha256(source_text):
            raise ValueError(f"Span source hash mismatch: {ref}")
        kind = row.get("span_kind")
        if kind not in ("whole", "sentence"):
            raise ValueError(f"Unknown span kind: {ref}")
        count = row.get("token_count")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(f"Invalid span token count: {ref}")
        if kind == "sentence" and count > max_tokens:
            raise ValueError(f"Span exceeds the token bound: {ref}")
        size = row.get("span_sentences")
        if isinstance(size, bool) or size not in (1, 2):
            raise ValueError(f"Invalid span sentence count: {ref}")
        chunk = row.get("chunk")
        if isinstance(chunk, bool) or not isinstance(chunk, int) or chunk < 0:
            raise ValueError(f"Invalid span chunk number: {ref}")
        match = SPAN_ID_RE.fullmatch(row["id"])
        surah, ayah = (int(part) for part in ref.split(":"))
        if match is None or match["field"] != row["field"] or int(match["surah"]) != surah or int(match["ayah"]) != ayah:
            raise ValueError(f"Span ID does not match its source parent: {ref}")
        if kind == "whole":
            if (
                match["suffix"] != "whole" or chunk != 0 or start != 0 or end != len(source_text)
                or row.get("sentence_count") != len(sentence_ranges(source_text))
            ):
                raise ValueError(f"Malformed whole-document row: {ref}")
        elif match["suffix"] == "whole" or chunk != int(match["chunk"]):
            raise ValueError(f"Malformed sentence chunk ID: {ref}")
        by_parent.setdefault(ref, []).append(row)
    for ref, parent_rows in by_parent.items():
        whole = [row for row in parent_rows if row["span_kind"] == "whole"]
        if len(whole) != 1 or whole[0]["text"] != source_texts[ref]:
            raise ValueError(f"Whole-document retention failed: {ref}")
        sentences = sorted((row for row in parent_rows if row["span_kind"] == "sentence"), key=lambda row: row["chunk"])
        if not sentences or [row["chunk"] for row in sentences] != list(range(1, len(sentences) + 1)):
            raise ValueError(f"Missing or nonsequential sentence spans: {ref}")
        sizes = {row["span_sentences"] for row in parent_rows}
        if len(sizes) != 1:
            raise ValueError(f"Inconsistent sentence grouping sizes: {ref}")
        size = next(iter(sizes))
        expected_groups = group_sentence_ranges(source_texts[ref], size)
        grouped: dict[int, list[dict[str, Any]]] = {}
        cursor = 0
        for row in sentences:
            sentence_start, sentence_end = row.get("sentence_start"), row.get("sentence_end")
            if (
                isinstance(sentence_start, bool) or isinstance(sentence_end, bool)
                or not isinstance(sentence_start, int) or not isinstance(sentence_end, int)
                or sentence_start < 1 or sentence_end < sentence_start
            ):
                raise ValueError(f"Malformed sentence range metadata: {ref}")
            if row["sentence_count"] != sentence_end - sentence_start + 1:
                raise ValueError(f"Malformed sentence range metadata: {ref}")
            if source_texts[ref][cursor:row["span_start"]].strip():
                raise ValueError(f"Sentence spans omit source text: {ref}")
            grouped.setdefault(sentence_start, []).append(row)
            cursor = row["span_end"]
        if source_texts[ref][cursor:].strip():
            raise ValueError(f"Sentence spans omit source text: {ref}")
        if set(grouped) != {group[2] for group in expected_groups}:
            raise ValueError(f"Sentence spans do not match deterministic groups: {ref}")
        for group_start, group_end, sentence_start, sentence_end in expected_groups:
            pieces = grouped[sentence_start]
            if (
                any(piece["sentence_end"] != sentence_end for piece in pieces)
                or pieces[0]["span_start"] != group_start or pieces[-1]["span_end"] != group_end
                or any(not group_start <= piece["span_start"] < piece["span_end"] <= group_end for piece in pieces)
            ):
                raise ValueError(f"Sentence span boundaries do not match deterministic groups: {ref}")
    if set(by_parent) != set(source_texts):
        raise ValueError("Span export does not cover every source parent")
    return {
        "passed": True,
        "rows": len(rows),
        "parents": len(by_parent),
        "whole_documents": sum(row["span_kind"] == "whole" for row in rows),
        "sentence_spans": sum(row["span_kind"] == "sentence" for row in rows),
        "max_span_tokens": max(row["token_count"] for row in rows if row["span_kind"] == "sentence"),
        "exact_source_substrings": True,
        "stable_unique_ids": True,
        "no_empty_spans": True,
        "whole_document_retained": True,
        "aggregation": "One maximum-scored result per parent ayah; spans do not add votes.",
    }
