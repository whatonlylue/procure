"""Coverage for cross-agent memories (doc_type document|memory).

Stdlib unittest only (no new harness). Run from the project root:

    .venv/bin/python -m unittest discover -s tests
"""
from __future__ import annotations

import tempfile
import unittest

from app import mcp_server
from app.config import Settings
from app.service import RAGService
from stub_embedder import StubEmbedder
from app.store import normalize_doc_type

MEM_TEXT = (
    "Agent memory: the lighthouse project uses RRF fusion with alpha 0.5. "
    "The north beacon was relamped in March. " * 10
)
DOC_TEXT = (
    "Harbor tide tables for April. High water at dawn near the east pier. "
    "Ferry departures follow the morning tide. " * 10
)


def _settings(tmp: str) -> Settings:
    s = Settings()
    s.data_dir = tmp
    return s


class DocTypeNormalizeTest(unittest.TestCase):
    def test_canonical_and_aliases(self) -> None:
        self.assertEqual(normalize_doc_type(None), "document")
        self.assertEqual(normalize_doc_type(""), "document")
        self.assertEqual(normalize_doc_type("document"), "document")
        self.assertEqual(normalize_doc_type("Documents"), "document")
        self.assertEqual(normalize_doc_type("memory"), "memory")
        self.assertEqual(normalize_doc_type("agent memories"), "memory")
        self.assertEqual(normalize_doc_type("agent-memory"), "memory")

    def test_filter_mode_empty_means_no_filter(self) -> None:
        self.assertIsNone(normalize_doc_type(None, allow_empty=True))
        self.assertIsNone(normalize_doc_type("", allow_empty=True))
        self.assertEqual(
            normalize_doc_type("memory", allow_empty=True), "memory")

    def test_invalid_raises(self) -> None:
        with self.assertRaises(ValueError):
            normalize_doc_type("nope")


class MemoryServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.svc = RAGService(_settings(self.tmp.name), embedder=StubEmbedder())
        mem = self.svc.ingest_text("memory: beacons", MEM_TEXT,
                                   ["project-x"], doc_type="memory")
        self.assertEqual(mem["status"], "ready")
        self.assertEqual(mem["doc_type"], "memory")
        self.mem_id = mem["doc_id"]
        doc = self.svc.ingest_text("tides.md", DOC_TEXT, ["project-x"])
        self.assertEqual(doc["status"], "ready")
        self.assertEqual(doc["doc_type"], "document")
        self.doc_id = doc["doc_id"]

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_list_filter(self) -> None:
        all_docs = self.svc.list_documents()
        self.assertEqual(len(all_docs), 2)
        mems = self.svc.list_documents(doc_type="memory")
        self.assertEqual([d["doc_id"] for d in mems], [self.mem_id])
        docs = self.svc.list_documents(doc_type="document")
        self.assertEqual([d["doc_id"] for d in docs], [self.doc_id])
        # Alias accepted.
        mems2 = self.svc.list_documents(doc_type="agent memories")
        self.assertEqual([d["doc_id"] for d in mems2], [self.mem_id])

    def test_search_type_filter(self) -> None:
        hits = self.svc.search("beacon relamped", doc_type="memory")
        self.assertTrue(hits)
        self.assertEqual({h["doc_id"] for h in hits}, {self.mem_id})
        self.assertEqual(hits[0]["doc_type"], "memory")
        # Scoped to documents, the memory is excluded (dense retrieval
        # still ranks the remaining document chunks).
        hits = self.svc.search("beacon relamped", doc_type="document")
        self.assertTrue(hits)
        self.assertNotIn(self.mem_id, {h["doc_id"] for h in hits})
        self.assertTrue(all(h["doc_type"] == "document" for h in hits))
        # Combined with tags (AND semantics).
        hits = self.svc.search("beacon tide", tags=["project-x"],
                               doc_type="memory")
        self.assertEqual({h["doc_id"] for h in hits}, {self.mem_id})

    def test_search_invalid_type_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.svc.search("beacon", doc_type="nope")

    def test_reingest_preserves_type(self) -> None:
        res = self.svc.ingest_text("memory: reingest.md", MEM_TEXT + "extra",
                                   doc_type="memory")
        self.assertEqual(res["status"], "ready")
        out = self.svc.reingest_document(res["doc_id"])
        self.assertEqual(out["status"], "ready")
        self.assertEqual(out["doc_type"], "memory")
        self.assertEqual(self.svc.meta.get(res["doc_id"])["doc_type"],
                         "memory")

    def test_reingest_extensionless_title(self) -> None:
        # Pasted-text titles like "memory: beacons" carry no extension;
        # extraction falls back to the raw copy's .md suffix.
        out = self.svc.reingest_document(self.mem_id)
        self.assertEqual(out["status"], "ready")
        self.assertEqual(out["doc_type"], "memory")
        doc = self.svc.get_document(self.mem_id)
        self.assertEqual(doc["text"], MEM_TEXT)

class MemoryMcpTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.svc = RAGService(_settings(self.tmp.name), embedder=StubEmbedder())
        res = self.svc.ingest_text("memory: beacons", MEM_TEXT,
                                   doc_type="memory")
        self.mem_id = res["doc_id"]
        self.svc.ingest_text("tides.md", DOC_TEXT)
        self._prev = mcp_server._service
        mcp_server._service = self.svc

    def tearDown(self) -> None:
        mcp_server._service = self._prev
        self.tmp.cleanup()

    def test_search_and_list_filters(self) -> None:
        hits = mcp_server.procure_search("beacon relamped",
                                         doc_type="memory")
        self.assertTrue(hits)
        self.assertEqual(hits[0].doc_id, self.mem_id)
        self.assertEqual(hits[0].doc_type, "memory")
        docs = mcp_server.procure_list_documents(doc_type="memory")
        self.assertEqual([d.doc_id for d in docs], [self.mem_id])
        self.assertEqual(docs[0].doc_type, "memory")

    def test_add_text_memory(self) -> None:
        res = mcp_server.procure_add_text(
            "memory: tides", "Memory: tides run high at dawn. " + DOC_TEXT,
            doc_type="memory")
        self.assertEqual(res.doc_type, "memory")
        doc = self.svc.get_document(res.doc_id)
        self.assertEqual(doc["doc_type"], "memory")

    def test_guide_mentions_memories(self) -> None:
        guide = mcp_server.read_guide()
        self.assertIn('doc_type="memory"', guide)
        self.assertIn("cross-agent", guide.lower())


if __name__ == "__main__":
    unittest.main()
