"""Ingestion progress streams during a document, not just between files.

Regression test for the hanging status bar: ingesting ONE large document
used to emit only progress(0, 1) at the start and progress(1, 1) at the
end, so the UI bar sat at 0% until the file finished. It must now emit
strictly-interior updates while the file is being embedded.

Run from the project root: .venv/bin/python -m unittest discover -s tests
"""
from __future__ import annotations

import sys
import os
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from app.config import Settings
from app.service import RAGService
from stub_embedder import StubEmbedder

TEXT = ("Procure keeps field notes about lighthouse maintenance. "
        "The north beacon was relamped in March. " * 60)


def _settings(tmp: str) -> Settings:
    s = Settings()
    s.data_dir = tmp
    return s


class IngestProgressTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.svc = RAGService(_settings(self.tmp.name),
                              embedder=StubEmbedder())
        # Force several embed batches so per-batch updates must fire.
        self.svc.embedder.batch_size = 4  # type: ignore[attr-defined]

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_single_file_emits_mid_ingest_updates(self) -> None:
        calls: list[tuple] = []
        res = self.svc.ingest_files(
            [("big.md", TEXT.encode("utf-8"))],
            progress=lambda d, t, c: calls.append((d, t, c)))
        self.assertEqual(res[0]["status"], "ready")
        self.assertGreaterEqual(len(calls), 3)
        # First call opens the file slot, last call closes the batch.
        self.assertEqual((calls[0][0], calls[0][1]), (0, 1))
        self.assertEqual((calls[-1][0], calls[-1][1]), (1, 1))
        # At least one update lands strictly inside the file's slot.
        mid = [c for c in calls[1:-1] if 0 < c[0] < 1]
        self.assertTrue(mid, f"no mid-ingest progress in {calls}")
        # Totals stay constant and done never runs backwards.
        self.assertTrue(all(c[1] == 1 for c in calls))
        dones = [c[0] for c in calls]
        self.assertEqual(dones, sorted(dones))

    def test_import_streams_per_document_progress(self) -> None:
        calls: list[tuple] = []
        payload = {"documents": [{
            "filename": "imported.md", "text": TEXT,
            "tags": [], "doc_type": "document",
        }]}
        res = self.svc.import_library(
            payload, progress=lambda d, t, c: calls.append((d, t, c)))
        self.assertEqual(res[0]["status"], "ready")
        mid = [c for c in calls[1:-1] if 0 < c[0] < 1] if len(calls) > 2 else []
        self.assertTrue(mid, f"no mid-ingest progress in {calls}")

    def test_reingest_streams_progress(self) -> None:
        res = self.svc.ingest_text("re.md", TEXT)
        doc_id = res["doc_id"]
        calls: list[tuple] = []
        out = self.svc.reingest_document(
            doc_id,
            progress_frac=lambda frac, stage: calls.append((frac, stage)))
        self.assertEqual(out["status"], "ready")
        self.assertGreaterEqual(len(calls), 2)
        fracs = [c[0] for c in calls]
        self.assertEqual(fracs, sorted(fracs))
        self.assertTrue(all(0.0 <= f < 1.0 for f in fracs))


if __name__ == "__main__":
    unittest.main()
