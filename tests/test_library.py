"""Coverage for new extractors, dedup, tags, and jobs.

Stdlib unittest only (no new harness). Run from the project root:

    .venv/bin/python -m unittest discover -s tests
"""
from __future__ import annotations

import os
import tempfile
import unittest
import zipfile
from unittest import mock

from app import extract
from app.config import Settings
from app.jobs import JobManager
from app.service import RAGService
from stub_embedder import StubEmbedder

TEXT_A = (
    "Lighthouse maintenance log. The north beacon was relamped in March. "
    "Spare lenses are stored in the cliff shed. " * 10
)
TEXT_B = (
    "Harbor tide tables for April. High water at dawn near the east pier. "
    "Ferry departures follow the morning tide. " * 10
)


def _settings(tmp: str) -> Settings:
    s = Settings()
    s.data_dir = tmp
    return s


def _write(tmp: str, name: str, data: bytes) -> str:
    path = os.path.join(tmp, name)
    with open(path, "wb") as f:
        f.write(data)
    return path


def _write_zip(path: str, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as z:
        for name, data in members.items():
            z.writestr(name, data)


XLSX = {
    "xl/workbook.xml": (
        b'<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        b'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        b'<sheets><sheet name="Tides" sheetId="1" r:id="rId1"/></sheets></workbook>'
    ),
    "xl/_rels/workbook.xml.rels": (
        b'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        b'<Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>'
    ),
    "xl/sharedStrings.xml": (
        b'<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        b"<si><t>Harbor</t></si><si><t>Apricot</t></si></sst>"
    ),
    "xl/worksheets/sheet1.xml": (
        b'<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        b"<sheetData>"
        b'<row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1"><v>42</v></c></row>'
        b'<row r="2"><c r="A2" t="s"><v>1</v></c>'
        b'<c r="B2" t="inlineStr"><is><t>fresh figs</t></is></c></row>'
        b"</sheetData></worksheet>"
    ),
}

EPUB = {
    "META-INF/container.xml": (
        b'<?xml version="1.0"?><container version="1.0" '
        b'xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
        b"<rootfiles>"
        b'<rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>'
        b"</rootfiles></container>"
    ),
    "OEBPS/content.opf": (
        b'<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" version="3.0">'
        b"<metadata><dc:title xmlns:dc=\"http://purl.org/dc/elements/1.1/\">"
        b"Beacon Tales</dc:title></metadata>"
        b"<manifest>"
        b'<item id="ch1" href="ch1.xhtml" media-type="application/xhtml+xml"/>'
        b'<item id="ch2" href="ch2.xhtml" media-type="application/xhtml+xml"/>'
        b"</manifest>"
        b"<spine><itemref idref=\"ch1\"/><itemref idref=\"ch2\"/></spine>"
        b"</package>"
    ),
    "OEBPS/ch1.xhtml": b"<html><body><h1>First</h1><p>Lighthouse lore begins.</p></body></html>",
    "OEBPS/ch2.xhtml": b"<html><body><p>Second chapter storms.</p></body></html>",
}

ODT = {
    "content.xml": (
        b'<?xml version="1.0"?><office:document-content '
        b'xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
        b'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0">'
        b"<office:body><office:text>"
        b"<text:h>Harbor Report</text:h>"
        b"<text:p>Tide tables attached below.</text:p>"
        b"</office:text></office:body></office:document-content>"
    ),
}

RTF = (
    b"{\\rtf1\\ansi{\\fonttbl{\\f0 Arial;}}"
    b"Harbor report\\par Tide tables \\b attached\\b0 below.}"
)


class ExtractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_csv_tsv(self) -> None:
        p = _write(self.tmp.name, "t.csv", b"port,tide\nHarbor,high\n")
        text = extract.extract_text(p, "t.csv")
        self.assertIn("Harbor", text)
        self.assertIn("port | tide", text)
        p = _write(self.tmp.name, "t.tsv", b"a\tb\n1\t2\n")
        self.assertIn("a | b", extract.extract_text(p, "t.tsv"))

    def test_xlsx_shared_inline_numeric(self) -> None:
        p = os.path.join(self.tmp.name, "t.xlsx")
        _write_zip(p, XLSX)
        text = extract.extract_text(p, "t.xlsx")
        self.assertIn("# Tides", text)
        self.assertIn("Harbor | 42", text)
        self.assertIn("Apricot | fresh figs", text)

    def test_epub_spine_order(self) -> None:
        p = os.path.join(self.tmp.name, "t.epub")
        _write_zip(p, EPUB)
        text = extract.extract_text(p, "t.epub")
        self.assertIn("Beacon Tales", text)
        self.assertLess(text.index("Lighthouse lore"), text.index("Second chapter"))

    def test_odt(self) -> None:
        p = os.path.join(self.tmp.name, "t.odt")
        _write_zip(p, ODT)
        text = extract.extract_text(p, "t.odt")
        self.assertIn("Harbor Report", text)
        self.assertIn("Tide tables", text)

    def test_rtf(self) -> None:
        p = _write(self.tmp.name, "t.rtf", RTF)
        text = extract.extract_text(p, "t.rtf")
        self.assertIn("Harbor report", text)
        self.assertIn("Tide tables", text)
        self.assertNotIn("fonttbl", text)
        self.assertNotIn("Arial", text)

    def test_json_xml_plain(self) -> None:
        p = _write(self.tmp.name, "t.json", b'{"port": "Harbor", "n": 3}')
        self.assertIn("Harbor", extract.extract_text(p, "t.json"))
        p = _write(self.tmp.name, "t.xml", b"<r><a>Hello</a></r>")
        self.assertIn("Hello", extract.extract_text(p, "t.xml"))
        p = _write(self.tmp.name, "t.rst", b"Hello\n=====\n\nBody here.")
        self.assertIn("Body here", extract.extract_text(p, "t.rst"))

    def test_legacy_office_guidance(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            extract.extract_text("/none/x.doc", "x.doc")
        self.assertIn(".docx", str(ctx.exception))
        with self.assertRaises(ValueError):
            extract.extract_text("/none/x.exe", "x.exe")

    def test_image_without_ocr_fails_helpfully(self) -> None:
        with mock.patch("app.ocr.available", return_value=False):
            with self.assertRaises(ValueError) as ctx:
                extract.extract_text("/none/x.png", "x.png")
        self.assertIn("OCR", str(ctx.exception))


class DedupTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.svc = RAGService(_settings(self.tmp.name), embedder=StubEmbedder())

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_same_bytes_ingested_once(self) -> None:
        data = ("Dedup probe text. " * 40).encode()
        first = self.svc.ingest_files([("a.txt", data)])[0]
        self.assertEqual(first["status"], "ready")
        second = self.svc.ingest_files([("b.txt", data)])[0]
        self.assertEqual(second["status"], "duplicate")
        self.assertEqual(second["doc_id"], first["doc_id"])
        self.assertEqual(len(self.svc.list_documents()), 1)

    def test_same_text_ingested_once(self) -> None:
        first = self.svc.ingest_text("n1", TEXT_A)
        second = self.svc.ingest_text("n2", TEXT_A)
        self.assertEqual(first["status"], "ready")
        self.assertEqual(second["status"], "duplicate")

    def test_different_content_ingests(self) -> None:
        self.assertEqual(self.svc.ingest_text("n1", TEXT_A)["status"], "ready")
        self.assertEqual(self.svc.ingest_text("n2", TEXT_B)["status"], "ready")


class TagsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.svc = RAGService(_settings(self.tmp.name), embedder=StubEmbedder())
        self.a = self.svc.ingest_text("a.md", TEXT_A, ["Memory", "boats"])["doc_id"]
        self.b = self.svc.ingest_text("b.md", TEXT_B, ["boats"])["doc_id"]

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_normalized_and_listed(self) -> None:
        docs = {d["doc_id"]: d for d in self.svc.list_documents()}
        self.assertEqual(docs[self.a]["tags"], ["boats", "memory"])
        counts = {t["tag"]: t["count"] for t in self.svc.list_tags()}
        self.assertEqual(counts, {"boats": 2, "memory": 1})

    def test_search_tag_filter_and_semantics(self) -> None:
        hits = self.svc.search("beacon relamped", tags=["boats"])
        self.assertTrue(hits)
        self.assertEqual(hits[0]["doc_id"], self.a)
        both = self.svc.search("beacon tide", tags=["boats"])
        self.assertEqual({h["doc_id"] for h in both}, {self.a, self.b})
        only_a = self.svc.search("beacon tide", tags=["boats", "memory"])
        self.assertEqual({h["doc_id"] for h in only_a}, {self.a})

    def test_search_unknown_tag_returns_nothing(self) -> None:
        self.assertEqual(self.svc.search("beacon", tags=["nope"]), [])

    def test_set_tags_replaces(self) -> None:
        out = self.svc.set_tags(self.a, ["Harbor"])
        self.assertEqual(out["tags"], ["harbor"])
        with self.assertRaises(KeyError):
            self.svc.set_tags("missing", ["x"])


class JobsTest(unittest.TestCase):
    def test_submit_runs_to_done(self) -> None:
        mgr = JobManager(max_workers=1)
        try:
            job = mgr.submit("test", "label", lambda h: {"n": 41 + 1})
            done = mgr.wait(job["job_id"], timeout=10)
            self.assertEqual(done["status"], "done")
            self.assertEqual(done["result"], {"n": 42})
        finally:
            mgr.shutdown()

    def test_cancel_unknown_and_finished(self) -> None:
        mgr = JobManager(max_workers=1)
        try:
            self.assertIsNone(mgr.cancel("nope"))
            job = mgr.submit("test", "label", lambda h: 1)
            done = mgr.wait(job["job_id"], timeout=10)
            self.assertEqual(mgr.cancel(done["job_id"])["status"], "done")
        finally:
            mgr.shutdown()

    def test_failure_captured(self) -> None:
        mgr = JobManager(max_workers=1)
        try:
            def boom(handle):
                raise ValueError("kaput")

            job = mgr.submit("test", "label", boom)
            done = mgr.wait(job["job_id"], timeout=10)
            self.assertEqual(done["status"], "failed")
            self.assertIn("kaput", done["error"])
        finally:
            mgr.shutdown()


if __name__ == "__main__":
    unittest.main()
