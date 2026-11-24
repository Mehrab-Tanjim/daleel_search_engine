from __future__ import annotations

import os
import re
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT / "evaluation")):
    if path not in sys.path:
        sys.path.insert(0, path)

import sentence_spans
import utils
from test_ai_representation import FakeDatabase, FakeDocument, deployment_with


class FakeTokenizer:
    def _tokens(self, text):
        return [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]

    def encode(self, text, add_special_tokens, truncation):
        assert add_special_tokens is True
        assert truncation is False
        return list(range(len(self._tokens(text)) + 2))

    def __call__(self, text, add_special_tokens, return_offsets_mapping, truncation):
        assert add_special_tokens is True
        assert return_offsets_mapping is True
        assert truncation is False
        return {
            "input_ids": list(range(len(self._tokens(text)) + 2)),
            "offset_mapping": [(0, 0), *self._tokens(text), (0, 0)],
        }


class SentenceSegmentationTests(unittest.TestCase):
    def test_abbreviations_quotes_and_parentheses_are_conservative(self):
        text = 'Dr. Smith said, "Do not leave (even now)." Then they left.'
        ranges = sentence_spans.sentence_ranges(text)
        self.assertEqual([text[start:end] for start, end in ranges], [
            'Dr. Smith said, "Do not leave (even now)."',
            "Then they left.",
        ])

    def test_colons_qualifiers_and_negation_are_preserved_exactly(self):
        text = "Rule: do not proceed without permission (even in emergencies). Stop only when safe."
        rows = sentence_spans.build_span_rows(
            text,
            {"surah": 2, "ayah": 3},
            "translation",
            "saheeh_international",
            1,
            FakeTokenizer(),
        )
        self.assertEqual(rows[0]["text"], text)
        self.assertEqual(rows[1]["text"], "Rule: do not proceed without permission (even in emergencies).")
        self.assertIn("do not", rows[1]["text"])
        self.assertIn("(even in emergencies)", rows[1]["text"])
        for row in rows:
            self.assertEqual(text[row["span_start"]:row["span_end"]], row["text"])

    def test_two_sentence_groups_use_original_offsets(self):
        text = "First sentence. Second sentence! Third sentence?"
        rows = sentence_spans.build_span_rows(
            text, {"surah": 1, "ayah": 1}, "translation", "saheeh_international", 2, FakeTokenizer()
        )
        self.assertEqual([row["text"] for row in rows[1:]], [
            "First sentence. Second sentence!",
            "Third sentence?",
        ])
        self.assertEqual([row["sentence_count"] for row in rows[1:]], [2, 1])
        self.assertEqual(rows[1]["span_start"], 0)
        self.assertEqual(rows[1]["span_end"], len("First sentence. Second sentence!"))

    def test_unicode_terminators_and_unterminated_tail_are_not_dropped(self):
        text = "هل هذا صحيح؟ نعم. unfinished tail"
        ranges = sentence_spans.sentence_ranges(text)
        self.assertEqual([text[start:end] for start, end in ranges], [
            "هل هذا صحيح؟", "نعم.", "unfinished tail",
        ])

    def test_empty_sources_are_rejected(self):
        for text in ("", " \n\t"):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, "empty source"):
                sentence_spans.build_span_rows(
                    text, {"surah": 1, "ayah": 1}, "translation", "saheeh_international", 1, FakeTokenizer()
                )

    def test_span_configuration_is_strict(self):
        self.assertEqual(sentence_spans.canonical_span_config("sentences", "2"), ("sentences", 2))
        for mode, count in (("other", 1), ("sentences", 0), ("sentences", 3), ("sentences", True), ("sentences", 1.5)):
            with self.subTest(mode=mode, count=count), self.assertRaises(ValueError):
                sentence_spans.canonical_span_config(mode, count)


class BoundedSpanTests(unittest.TestCase):
    def test_long_sentence_is_split_into_bounded_exact_source_pieces(self):
        text = " ".join(f"word{index}" for index in range(31)) + "."
        rows = sentence_spans.build_span_rows(
            text, {"surah": 2, "ayah": 255}, "translation", "saheeh_international", 1, FakeTokenizer(), max_tokens=8
        )
        spans = [row for row in rows if row["span_kind"] == "sentence"]
        self.assertGreater(len(spans), 1)
        self.assertTrue(all(row["token_count"] <= 8 for row in spans))
        self.assertEqual(" ".join(row["text"] for row in spans), text)
        self.assertTrue(all(text[row["span_start"]:row["span_end"]] == row["text"] for row in spans))

    def test_prefix_without_source_capacity_is_rejected(self):
        text = "x " * 20
        with self.assertRaisesRegex(ValueError, "leaves no capacity"):
            sentence_spans.build_span_rows(
                text, {"surah": 1, "ayah": 1}, "translation", "saheeh_international", 1, FakeTokenizer(), max_tokens=3
            )

    def test_audit_checks_grounding_ids_bounds_and_whole_retention(self):
        text = "First sentence. Second sentence."
        rows = sentence_spans.build_span_rows(
            text, {"surah": 1, "ayah": 1}, "translation", "saheeh_international", 2, FakeTokenizer()
        )
        audit = sentence_spans.audit_span_rows(rows, {"1:1": text})
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["whole_documents"], 1)
        self.assertEqual(audit["sentence_spans"], 1)
        corrupted = [dict(row) for row in rows]
        corrupted[1]["text"] = "invented"
        with self.assertRaisesRegex(ValueError, "exact source substring"):
            sentence_spans.audit_span_rows(corrupted, {"1:1": text})
        duplicated = rows + [dict(rows[-1])]
        with self.assertRaisesRegex(ValueError, "duplicated"):
            sentence_spans.audit_span_rows(duplicated, {"1:1": text})

    def test_ids_are_stable_across_repeated_exports(self):
        text = "First sentence. Second sentence."
        metadata = {"surah": 1, "ayah": 1}
        first = sentence_spans.build_span_rows(text, metadata, "translation", "saheeh_international", 1, FakeTokenizer())
        second = sentence_spans.build_span_rows(text, metadata, "translation", "saheeh_international", 1, FakeTokenizer())
        self.assertEqual([row["id"] for row in first], [row["id"] for row in second])
        self.assertEqual(first[0]["id"], "translation:1:1:whole")
        self.assertEqual(first[1]["id"], "translation:1:1:s0001")


class DocumentExpansionTests(unittest.TestCase):
    def test_expansion_supports_legacy_metadata_and_retains_whole_first(self):
        from langchain_core.documents import Document

        document = Document(
            page_content="cleaned text",
            metadata={"SurahNo": "1", "AyahNo": "2", "original_text": "Original: do not omit (this)."},
        )
        expanded = sentence_spans.expand_documents([document], 1, FakeTokenizer())
        self.assertEqual(expanded[0].page_content, "passage: Original: do not omit (this).")
        self.assertEqual(expanded[0].metadata["span_kind"], "whole")
        self.assertEqual(expanded[1].metadata["span_kind"], "sentence")
        self.assertEqual(expanded[1].metadata["surah"], 1)
        self.assertEqual(expanded[1].metadata["ayah"], 2)
        self.assertIn("(this)", expanded[1].page_content)


class SpanRetrievalTests(unittest.TestCase):
    def test_multiple_spans_collapse_to_one_max_scored_ayah(self):
        first = FakeDocument(1, 1, text="first span", score_id="translation:1:1:s0001")
        second = FakeDocument(1, 1, text="second span", score_id="translation:1:1:s0002")
        third = FakeDocument(1, 2, text="other ayah", score_id="translation:1:2:s0001")
        first.metadata.update({"span_kind": "sentence", "span_start": 0, "span_end": 11})
        second.metadata.update({"span_kind": "sentence", "span_start": 12, "span_end": 24})
        deployment = deployment_with([(first, 0.7), (second, 0.9), (third, 0.8)])
        deployment.span_mode = "sentences"
        deployment.span_sentences = 1
        results = deployment.search("best_match", "query", k=2)
        self.assertEqual([doc.id for doc, _ in results], [second.id, third.id])
        self.assertEqual(len({doc.metadata["ayah"] for doc, _ in results}), 2)

    def test_winning_translation_span_displays_retained_whole_verse(self):
        whole = FakeDocument(1, 1, text="passage: Complete verse.", score_id="translation:1:1:whole")
        whole.metadata["span_kind"] = "whole"
        span = FakeDocument(1, 1, text="passage: Complete", score_id="translation:1:1:s0001")
        span.metadata.update({"span_kind": "sentence", "span_start": 0, "span_end": 8})
        deployment = deployment_with([(whole, 0.8), (span, 0.9)])
        deployment.span_mode = "sentences"
        deployment.span_sentences = 1
        deployment.translation_docs_by_ayah = {"1:1": whole}
        self.assertIs(deployment.display_document(span), whole)
        self.assertEqual(span.metadata["span_kind"], "sentence")
        self.assertEqual(span.page_content, "passage: Complete")

    def test_environment_switch_defaults_off_and_validates_values(self):
        database = FakeDatabase([])
        with mock.patch.dict(os.environ, {"SPAN_MODE": "invalid"}, clear=False):
            with mock.patch.object(utils, "HuggingFaceEmbeddings"):
                with self.assertRaisesRegex(ValueError, "span_mode"):
                    utils.VectorSearchDeployment("unused", "unused", "cpu")
        with mock.patch.dict(os.environ, {"SPAN_MODE": "sentences", "SPAN_SENTENCES": "3"}, clear=False):
            with mock.patch.object(utils, "HuggingFaceEmbeddings"):
                with self.assertRaisesRegex(ValueError, "span_sentences"):
                    utils.VectorSearchDeployment("unused", "unused", "cpu")
        with mock.patch.dict(os.environ, {"SPAN_MODE": "sentences", "SPAN_SENTENCES": "2"}, clear=False):
            with mock.patch.object(utils, "HuggingFaceEmbeddings"):
                with mock.patch.object(utils.FAISS, "load_local", return_value=database):
                    deployment = utils.VectorSearchDeployment("unused", "unused", "cpu")
        self.assertEqual(deployment.span_mode, "sentences")
        self.assertEqual(deployment.span_sentences, 2)


if __name__ == "__main__":
    unittest.main()
