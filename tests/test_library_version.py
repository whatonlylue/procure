"""Coverage for the cheap library revision fingerprint polled by the UI.

The Library tab only reloads on mount and after its own mutations, so
writes from the MCP server (a separate process sharing the data dir)
used to stay invisible until a manual refresh. The UI polls
GET /api/library/version and reloads when the fingerprint moves, so
every external mutation path must move it.

Stdlib unittest only (no new harness). Run from the project root:

    .venv/bin/python -m unittest discover -s tests
"""
from __future__ import annotations

import tempfile
import unittest
from unittest import mock

from app.config import Settings
from app.service import RAGService

TEXT_A = (
    "Lighthouse maintenance log. The north beacon was relamped in March. "
    "Spare lenses are stored in the cliff shed. " * 10
)
TEXT_B = (
    "Harbor tide tables for April. High water at dawn near the east pier. "
    "Ferry departures follow the morning tide. " * 10
)


def _svc(tmp: str) -> RAGService:
    s = Settings()
    s.data_dir = tmp
    return RAGService(s)


class LibraryVersionTest(unittest.TestCase):
    def test_empty_library_has_stable_version(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        svc = _svc(tmp.name)
        v = svc.library_version()
        self.assertEqual(
            v, {"documents": 0, "chunks": 0, "latest": "",
                "tags": 0, "tag_chars": 0})
        self.assertEqual(v, svc.library_version())

    def test_every_external_mutation_moves_version(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        svc = _svc(tmp.name)
        seen = {tuple(sorted(svc.library_version().items()))}

        def changed() -> None:
            key = tuple(sorted(svc.library_version().items()))
            self.assertNotIn(key, seen, "mutation left the version unchanged")
            seen.add(key)

        # MCP procure_add_text.
        res = svc.ingest_text("memory: beacons.md", TEXT_A,
                              ["session"], "memory")
        self.assertEqual(res["status"], "ready")
        changed()
        # Duplicate re-ingest with a new tag: no new document, but the
        # tag merge must still move the fingerprint.
        dup = svc.ingest_text("memory: beacons.md", TEXT_A,
                              ["session", "extra"], "memory")
        self.assertEqual(dup["status"], "duplicate")
        changed()
        # MCP procure_add_url / second memory.
        res_b = svc.ingest_text("memory: tides.md", TEXT_B, [], "memory")
        changed()
        # MCP procure_update_text (same doc, rewritten content).
        svc.update_document_text(res_b["doc_id"], TEXT_A + TEXT_B)
        changed()
        # Tag-only edit.
        svc.set_tags(res_b["doc_id"], ["harbor"])
        changed()
        # MCP procure_delete_document.
        self.assertTrue(svc.delete_document(res["doc_id"]))
        changed()

    def test_version_endpoint_matches_service(self) -> None:
        from starlette.testclient import TestClient

        from app.main import app

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        svc = _svc(tmp.name)
        with mock.patch("app.main.get_service", return_value=svc):
            client = TestClient(app, raise_server_exceptions=False)
            before = client.get("/api/library/version")
            self.assertEqual(before.status_code, 200)
            self.assertEqual(before.json(), svc.library_version())
            svc.ingest_text("memory: beacons.md", TEXT_A, [], "memory")
            after = client.get("/api/library/version")
            self.assertEqual(after.status_code, 200)
            self.assertEqual(after.json(), svc.library_version())
            self.assertNotEqual(before.json(), after.json())


if __name__ == "__main__":
    unittest.main()
