"""Focused coverage for document fetch + runtime retrieval settings.

Stdlib unittest only (no new harness). Run from the project root:

    .venv/bin/python -m unittest discover -s tests
"""
from __future__ import annotations

import os
import tempfile
import unittest

from app import mcp_server
from app.config import Settings
from app.service import RAGService

TEXT = (
    "Procure keeps field notes about lighthouse maintenance. "
    "The north beacon was relamped in March. " * 20
)


def _settings(tmp: str) -> Settings:
    s = Settings()
    s.data_dir = tmp
    return s


class DocumentFetchTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.svc = RAGService(_settings(self.tmp.name))
        res = self.svc.ingest_text("lighthouse.md", TEXT)
        self.assertEqual(res["status"], "ready")
        self.doc_id = res["doc_id"]

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_get_document_returns_full_text(self) -> None:
        doc = self.svc.get_document(self.doc_id)
        self.assertEqual(doc["doc_id"], self.doc_id)
        self.assertEqual(doc["filename"], "lighthouse.md")
        self.assertEqual(doc["text"], TEXT)
        self.assertGreater(doc["chunk_count"], 1)

    def test_get_document_unknown_raises(self) -> None:
        with self.assertRaises(KeyError):
            self.svc.get_document("nope-nope-nope")

    def test_get_document_falls_back_to_chunks(self) -> None:
        for name in os.listdir(os.path.join(self.tmp.name, "raw")):
            os.remove(os.path.join(self.tmp.name, "raw", name))
        doc = self.svc.get_document(self.doc_id)
        # Chunk overlap duplicates words, so compare coverage, not equality.
        for word in ("lighthouse", "relamped", "March", "beacon"):
            self.assertIn(word, doc["text"])

    def test_vector_store_get_by_doc_order(self) -> None:
        rows = self.svc.vectors.get_by_doc(self.doc_id)
        self.assertEqual(len(rows), self.svc.get_document(self.doc_id)["chunk_count"])
        idx = [int(cid.rsplit(":", 1)[1]) for cid, _ in rows]
        self.assertEqual(idx, sorted(idx))


class SettingsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.svc = RAGService(_settings(self.tmp.name))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_update_round_trip(self) -> None:
        self.assertEqual(self.svc.update_settings("dense", "off"),
                         {"search": "dense", "rerank": "off"})
        self.assertFalse(self.svc.settings.hybrid)
        self.assertFalse(self.svc.settings.rerank_enabled)
        self.assertEqual(self.svc.update_settings("hybrid", "on"),
                         {"search": "hybrid", "rerank": "on"})
        self.assertTrue(self.svc.settings.hybrid)

    def test_update_rejects_unknown(self) -> None:
        with self.assertRaises(ValueError):
            self.svc.update_settings("fuzzy", None)
        with self.assertRaises(ValueError):
            self.svc.update_settings(None, "sometimes")


class McpResourceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.svc = RAGService(_settings(self.tmp.name))
        res = self.svc.ingest_text("memory: beacons", TEXT)
        self.doc_id = res["doc_id"]
        self._prev = mcp_server._service
        mcp_server._service = self.svc

    def tearDown(self) -> None:
        mcp_server._service = self._prev
        self.tmp.cleanup()

    def test_read_document(self) -> None:
        body = mcp_server.read_document(self.doc_id)
        self.assertIn("memory: beacons", body)
        self.assertIn("relamped", body)

    def test_read_document_unknown(self) -> None:
        with self.assertRaises(ValueError):
            mcp_server.read_document("missing-doc")

    def test_read_guide(self) -> None:
        guide = mcp_server.read_guide()
        self.assertIn("procure_search", guide)
        self.assertIn("procure://documents/{doc_id}", guide)

    def test_tool_results_carry_doc_uri(self) -> None:
        hits = mcp_server.procure_search("lighthouse relamped")
        self.assertTrue(hits)
        self.assertEqual(hits[0].uri, f"procure://documents/{self.doc_id}")
        docs = mcp_server.procure_list_documents()
        self.assertEqual(docs[0].uri, f"procure://documents/{self.doc_id}")


if __name__ == "__main__":
    unittest.main()
