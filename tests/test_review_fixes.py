"""Regression coverage for the 2026-09-30 procure code review fixes.

Stdlib unittest only (no new harness). Run from the project root:

    .venv/bin/python -m unittest discover -s tests
"""
from __future__ import annotations

import os
import tempfile
import unittest
import zipfile
from unittest import mock

from app import chunking, extract, mcp_server, skills
from app.config import Settings
from app.jobs import JobManager
from app.search import (clear_answerability, combine_answerability,
                        coverage_score, sigmoid)
from app.service import RAGService
from stub_embedder import StubEmbedder
from app.store import MetaStore
from app.vectordb.sqlite_vec import DimensionMismatchError, SqliteVectorStore

LONG_A = ("Beaconuesten maintenance log. The north beacon was relamped. " * 40)
LONG_B = ("Harbor tide tables for April. Ferries follow the morning tide. " * 40)


def _settings(tmp: str, **kw) -> Settings:
    s = Settings()
    s.data_dir = tmp
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def _svc(tmp: str, embedder=None, **kw) -> RAGService:
    return RAGService(
        _settings(tmp, **kw),
        embedder=embedder if embedder is not None else StubEmbedder())


class StaleChunksTest(unittest.TestCase):
    def test_update_replaces_chunks(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        svc = _svc(tmp.name)
        res = svc.ingest_text("big.md", LONG_A)
        self.assertEqual(res["status"], "ready")
        self.assertGreater(res["chunk_count"], 1)
        doc_id = res["doc_id"]
        short = "A tiny replacement note about sparrows. " * 4
        out = svc.update_document_text(doc_id, short)
        self.assertEqual(out["status"], "ready")
        self.assertEqual(out["chunk_count"], 1)
        rows = svc.vectors.get_by_doc(doc_id)
        self.assertEqual(len(rows), 1)
        # Old content no longer searchable under this doc.
        # Scoped search ranks in-scope chunks even without term matches,
        # but no stale content may surface.
        hits = svc.search("relamped beacon", doc_ids=[doc_id])
        self.assertTrue(all("relamped" not in h["text"] for h in hits))
        self.assertTrue(all(h["chunk_id"] == f"{doc_id}:0" for h in hits))
        self.assertIn("sparrows", svc.get_document(doc_id)["text"])

    def test_reingest_failure_preserves_provenance(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        svc = _svc(tmp.name)
        res = svc.ingest_files([("w.txt", LONG_A.encode("utf-8"))])[0]
        doc_id = res["doc_id"]
        with mock.patch("app.extract.extract_text",
                        side_effect=RuntimeError("boom")):
            out = svc.reingest_document(doc_id)
        self.assertEqual(out["status"], "failed")
        doc = svc.meta.get(doc_id)
        self.assertEqual(doc["filename"], "w.txt")
        self.assertEqual(doc["doc_type"], "document")
        self.assertTrue(doc["content_hash"])


class DedupScopeTest(unittest.TestCase):
    def test_same_type_dedups_and_merges_tags(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        svc = _svc(tmp.name)
        first = svc.ingest_text("a.md", LONG_A, ["x"])
        second = svc.ingest_text("b.md", LONG_A, ["y"])
        self.assertEqual(second["status"], "duplicate")
        self.assertEqual(second["doc_id"], first["doc_id"])
        self.assertEqual(sorted(svc.meta.get_tags(first["doc_id"])), ["x", "y"])

    def test_memory_and_document_coexist(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        svc = _svc(tmp.name)
        doc = svc.ingest_text("a.md", LONG_A)
        mem = svc.ingest_text("m", LONG_A, doc_type="memory")
        self.assertEqual(mem["status"], "ready")
        self.assertNotEqual(mem["doc_id"], doc["doc_id"])
        self.assertEqual(len(svc.list_documents()), 2)

    def test_extensionless_title_gains_md(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        svc = _svc(tmp.name)
        res = svc.ingest_text("field note", LONG_A)
        self.assertEqual(res["filename"], "field note.md")


class ChunkingFixTest(unittest.TestCase):
    def test_overlap_never_duplicates_or_overflows(self) -> None:
        a = "A" * 880 + "."
        b = "B" * 200 + "."
        c = "C" * 100 + "."
        chunks = chunking.split_text(f"{a} {b} {c}", 1000, 150)
        for ch in chunks:
            self.assertLessEqual(len(ch), 1000)
        joined = " ".join(chunks)
        self.assertEqual(joined.count("A"), 880)

    def test_abbreviations(self) -> None:
        sents = chunking._split_sentences("No. Go. Hi. Mr. Smith came.")
        self.assertEqual(len(sents), 4)
        self.assertTrue(sents[3].startswith("Mr. Smith"))


class ExtractFixTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _write(self, name: str, data: bytes) -> str:
        path = os.path.join(self.tmp.name, name)
        with open(path, "wb") as f:
            f.write(data)
        return path

    def test_docx_unescapes_entities(self) -> None:
        path = os.path.join(self.tmp.name, "t.docx")
        with zipfile.ZipFile(path, "w") as z:
            z.writestr("word/document.xml",
                       "<w:document><w:body><w:p><w:r><w:t>Tom &amp; Jerry &#8217;s</w:t>"
                       "</w:r></w:p></w:body></w:document>")
        text = extract.extract_text(path, "t.docx")
        self.assertIn("Tom & Jerry \u2019s", text)

    def test_pptx_numeric_slide_order(self) -> None:
        path = os.path.join(self.tmp.name, "t.pptx")
        with zipfile.ZipFile(path, "w") as z:
            for i in (1, 2, 10):
                z.writestr(f"ppt/slides/slide{i}.xml",
                           f"<p:sld><p:txBody><a:t>SLIDE{i}</a:t></p:txBody></p:sld>")
        text = extract.extract_text(path, "t.pptx")
        self.assertLess(text.index("SLIDE1"), text.index("SLIDE2"))
        self.assertLess(text.index("SLIDE2"), text.index("SLIDE10"))

    def test_odt_heading_prefixes_heading(self) -> None:
        path = os.path.join(self.tmp.name, "t.odt")
        with zipfile.ZipFile(path, "w") as z:
            z.writestr("content.xml",
                       "<office:document-content><office:body><office:text>"
                       "<text:h>Harbor Report</text:h>"
                       "<text:p>Tide tables below.</text:p>"
                       "</office:text></office:body></office:document-content>")
        text = extract.extract_text(path, "t.odt")
        self.assertTrue(text.startswith("# Harbor Report"))

    def test_html_drops_script_style(self) -> None:
        path = self._write("t.html", b"<html><head><style>.x{color:red}</style></head>"
                                     b"<body><script>var secret=1;</script>"
                                     b"<p>Hi there visible text for the library index.</p>"
                                     b"</body></html>")
        text = extract.extract_text(path, "t.html")
        self.assertNotIn("secret", text)
        self.assertNotIn("color", text)
        self.assertIn("Hi there", text)


class RankingBoundsTest(unittest.TestCase):
    def test_answerability_stays_in_unit_interval(self) -> None:
        for a in clear_answerability([0.0, 10.0, -1000.0, 1000.0],
                                     [0.0, 1.0, 0.5, 0.0], alpha=0.5):
            self.assertGreaterEqual(a, 0.0)
            self.assertLessEqual(a, 1.0)
        for a in combine_answerability([100.0, -100.0, 0.0], [0.5, 0.5, 0.5]):
            self.assertGreaterEqual(a, 0.0)
            self.assertLessEqual(a, 1.0)
        # Clamped: no OverflowError, saturates to within a hair of 0/1.
        lo, hi = sigmoid([-1000.0, 1000.0])
        self.assertLess(lo, 1e-200)
        self.assertEqual(hi, 1.0)

    def test_phrase_bonus_uses_raw_bigrams(self) -> None:
        score = coverage_score("capital of france",
                               "The capital of the country is unknown.")
        # base 0.5 (capital hit, france miss) + 0.25 * 1/2 phrase bonus.
        self.assertAlmostEqual(score, 0.625)


class VectorDimTest(unittest.TestCase):
    def test_ragged_batch_raises(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store = SqliteVectorStore(os.path.join(tmp.name, "v.sqlite"))
        with self.assertRaises(DimensionMismatchError):
            store.upsert(["a", "b"], ["d", "d"],
                         [[1.0, 2.0], [1.0, 2.0, 3.0]], ["x", "y"])

    def test_dim_change_is_reported_not_silent(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        svc = _svc(tmp.name)
        svc.ingest_text("a.md", LONG_A)
        status = svc.dimension_status()
        self.assertTrue(status["dense_ok"])
        # A different embedding backend over the same store must be
        # reported, not silently served: dense is skipped, sparse answers.
        other = _svc(tmp.name, embedder=StubEmbedder(backend="other-stub"))
        status = other.dimension_status()
        self.assertFalse(status["dense_ok"])
        self.assertIn("warning", status)
        hits = other.search("beacon")
        self.assertTrue(hits)  # sparse channel still answers
        self.assertTrue(all(h["dense_score"] == 0.0 for h in hits))
        self.assertGreater(other.vectors.last_skipped, 0)

    def test_legacy_untagged_corpus_is_reported_not_silent(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        svc = _svc(tmp.name)
        svc.ingest_text("a.md", LONG_A)
        # Simulate a pre-tag corpus (the hash era): vectors present, no
        # backend tag. Same width must still NOT read as healthy.
        with svc.vectors._session() as c:
            c.execute("DELETE FROM store_meta WHERE key='backend'")
        self.assertIsNone(svc.vectors.stored_backend())
        status = svc.dimension_status()
        self.assertFalse(status["dense_ok"])
        self.assertIn("legacy", status["warning"])


class StoreFixTest(unittest.TestCase):
    def test_update_status_preserves_provenance(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        meta = MetaStore(os.path.join(tmp.name, "m.sqlite"))
        meta.upsert("d1", "f.md", "ready", 3, "", "abc")
        meta.update_status("d1", "processing")
        doc = meta.get("d1")
        self.assertEqual(doc["status"], "processing")
        self.assertEqual(doc["content_hash"], "abc")
        self.assertEqual(doc["filename"], "f.md")

    def test_list_filters_and_paging(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        svc = _svc(tmp.name)
        svc.ingest_text("apple.md", LONG_A, ["fruit"])
        svc.ingest_text("memory: boats", LONG_B, ["sea"], "memory")
        self.assertEqual(svc.count_documents(query="apple"), 1)
        self.assertEqual(len(svc.list_documents(doc_type="memory")), 1)
        page = svc.list_documents(limit=1, offset=0, sort="name")
        self.assertEqual(len(page), 1)
        self.assertEqual(page[0]["filename"], "apple.md")
        self.assertEqual(svc.count_documents(), 2)

    def test_since_validation(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        meta = MetaStore(os.path.join(tmp.name, "m.sqlite"))
        with self.assertRaises(ValueError):
            meta.doc_ids_created_since("yesterday")


class JobsPruneTest(unittest.TestCase):
    def test_active_jobs_survive_pruning(self) -> None:
        mgr = JobManager(max_workers=1)
        self.addCleanup(mgr.shutdown)
        for i in range(5):
            jid = f"done-{i}"
            mgr._jobs[jid] = {"job_id": jid, "status": "done"}
            mgr._order.append(jid)
        mgr._jobs["active-1"] = {"job_id": "active-1", "status": "running"}
        mgr._order.append("active-1")
        mgr._futures["active-1"] = mock.Mock()
        import app.jobs as jobs_mod
        orig = jobs_mod._MAX_JOBS
        jobs_mod._MAX_JOBS = 3
        try:
            mgr._prune_locked()
        finally:
            jobs_mod._MAX_JOBS = orig
        remaining = [j for j in mgr._order if j in mgr._jobs]
        self.assertIn("active-1", remaining)
        self.assertLessEqual(len(remaining), 3)
        # Order of survivors is stable (no active job rotated to the end).
        self.assertEqual(remaining[-1], "active-1")


class MetaUpdateExportTest(unittest.TestCase):
    def test_update_text_and_meta(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        svc = _svc(tmp.name)
        res = svc.ingest_text("note.md", LONG_A, doc_type="memory")
        doc_id = res["doc_id"]
        out = svc.update_document_text(doc_id, LONG_B, tags=["new"])
        self.assertEqual(out["status"], "ready")
        self.assertEqual(out["doc_type"], "memory")
        self.assertEqual(out["tags"], ["new"])
        self.assertIn("Ferries", svc.get_document(doc_id)["text"])
        renamed = svc.update_document_meta(doc_id, filename="renamed.md",
                                           doc_type="document")
        self.assertEqual(renamed["filename"], "renamed.md")
        self.assertEqual(renamed["doc_type"], "document")
        with self.assertRaises(KeyError):
            svc.update_document_meta("nope", filename="x")

    def test_export_import_roundtrip(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        svc = _svc(tmp.name)
        svc.ingest_text("memory: a", LONG_A, ["t"], "memory")
        payload = svc.export_library()
        self.assertEqual(len(payload["documents"]), 1)
        tmp2 = tempfile.TemporaryDirectory()
        self.addCleanup(tmp2.cleanup)
        svc2 = _svc(tmp2.name)
        out = svc2.import_library(payload)
        self.assertEqual(out[0]["status"], "ready")
        self.assertEqual(svc2.list_documents()[0]["doc_type"], "memory")
        self.assertIn("relamped",
                      svc2.search("relamped")[0]["text"])


class LocalOnlyMiddlewareTest(unittest.TestCase):
    def _client(self):
        from starlette.applications import Starlette
        from starlette.responses import PlainTextResponse
        from starlette.routing import Route
        from starlette.testclient import TestClient

        from app.main import LocalOnlyMiddleware

        async def home(_request):
            return PlainTextResponse("ok")

        inner = Starlette(routes=[Route("/", home)])
        inner.add_middleware(LocalOnlyMiddleware)
        return TestClient(inner, raise_server_exceptions=False)

    def test_same_origin_passes_with_csp(self) -> None:
        r = self._client().get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("default-src 'self'", r.headers["content-security-policy"])

    def test_foreign_host_rejected(self) -> None:
        r = self._client().get("/", headers={"host": "evil.example:8000"})
        self.assertEqual(r.status_code, 403)

    def test_cross_site_origin_rejected(self) -> None:
        r = self._client().post("/", headers={"origin": "http://evil.example"})
        self.assertEqual(r.status_code, 403)


class SkillsFixTest(unittest.TestCase):
    def test_frontmatter_keeps_spaces(self) -> None:
        text = "---\nname: procure\ndescription: a long human sentence here\n---\n"
        self.assertEqual(skills.frontmatter_field(text, "description"),
                         "a long human sentence here")

    def test_staged_copy_matches_source(self) -> None:
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        self.assertEqual(
            (root / "skills" / "procure" / "SKILL.md").read_bytes(),
            (root / "app" / "_skill_data" / "SKILL.md").read_bytes())

    def test_force_update_preserves_extras(self) -> None:
        import os

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        with mock.patch.dict(os.environ, {"HOME": tmp.name}):
            skills.install(["claude"])
            dest = skills.get_target("claude").path()
            extra = dest / "MYNOTES.md"
            extra.write_text("mine", encoding="utf-8")
            out = skills.install(["claude"], force=True)
            self.assertEqual(out["results"][0]["action"], "updated")
            self.assertTrue(extra.is_file())
            self.assertEqual(extra.read_text(encoding="utf-8"), "mine")

    def test_uninstall_keeps_extras(self) -> None:
        import os

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        with mock.patch.dict(os.environ, {"HOME": tmp.name}):
            skills.install(["agents"])
            dest = skills.get_target("agents").path()
            (dest / "EXTRA.md").write_text("x", encoding="utf-8")
            out = skills.uninstall(["agents"])
            self.assertEqual(out["results"][0]["action"], "removed")
            self.assertFalse((dest / "SKILL.md").exists())
            self.assertTrue((dest / "EXTRA.md").is_file())


class McpLifecycleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.svc = _svc(self.tmp.name)
        res = self.svc.ingest_text("memory: a", LONG_A, doc_type="memory")
        self.mem_id = res["doc_id"]
        self._prev = mcp_server._service
        mcp_server._service = self.svc

    def tearDown(self) -> None:
        mcp_server._service = self._prev

    def test_update_and_delete(self) -> None:
        updated = mcp_server.procure_update_text(self.mem_id, LONG_B)
        self.assertEqual(updated.status, "ready")
        self.assertIn("Ferries",
                      self.svc.get_document(self.mem_id)["text"])
        with self.assertRaises(ValueError):
            mcp_server.procure_update_text("missing", LONG_B)
        docs = mcp_server.procure_list_documents(limit=1)
        self.assertEqual(len(docs), 1)
        out = mcp_server.procure_delete_document(self.mem_id)
        self.assertEqual(out, {"deleted": self.mem_id})
        with self.assertRaises(ValueError):
            mcp_server.procure_delete_document(self.mem_id)

    def test_tool_descriptions_cover_new_tools(self) -> None:
        names = {t["name"] for t in mcp_server.tool_descriptions()}
        self.assertIn("procure_update_text", names)
        self.assertIn("procure_delete_document", names)


class CliTest(unittest.TestCase):
    def setUp(self) -> None:
        self._env = dict(os.environ)

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._env)

    def test_version(self) -> None:
        from app import cli
        from app.version import __version__

        with self.assertRaises(SystemExit) as ctx:
            cli.main(["--version"])
        self.assertEqual(ctx.exception.code, 0)
        self.assertTrue(__version__)

    def test_add_list_search(self) -> None:
        from unittest import mock

        from app import cli

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = os.path.join(tmp.name, "note.txt")
        with open(path, "w") as f:
            f.write(LONG_A)
        # The CLI builds its own RAGService; keep this hermetic (no model
        # download) by substituting the deterministic test embedder.
        with mock.patch("app.service.Model2VecEmbedder",
                        lambda *a, **k: StubEmbedder()):
            self.assertEqual(
                cli.main(["add", path, "--data-dir",
                          os.path.join(tmp.name, "data")]), 0)
            data = os.path.join(tmp.name, "data")
            self.assertEqual(cli.main(["list", "--data-dir", data]), 0)
            self.assertEqual(
                cli.main(["search", "relamped", "--data-dir", data]), 0)


class SettingsPersistTest(unittest.TestCase):
    def test_overrides_survive_restart(self) -> None:
        import os

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        data = os.path.join(tmp.name, "data")
        svc = _svc(data)
        svc.update_settings("dense", "off")
        svc.set_mcp_autostart(True)
        with mock.patch.dict(os.environ, {"PROCURE_DATA_DIR": data}):
            fresh = Settings()
        self.assertEqual(fresh.search_mode, "dense")
        self.assertEqual(fresh.rerank, "off")
        self.assertTrue(fresh.mcp_autostart)


if __name__ == "__main__":
    unittest.main()
