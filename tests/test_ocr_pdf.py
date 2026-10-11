import os
import stat
import tempfile
import unittest
from unittest import mock

import pypdfium2 as pdfium

from app import ocr
from app.extract import extract_text


def make_pdf(pages: list[str], size: int = 12) -> bytes:
    """Minimal hand-written PDF: one Helvetica text line per page."""
    objs: list[bytes] = [b"<< /Type /Catalog /Pages 2 0 R >>"]
    n = len(pages)
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(n))
    objs.append(f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode())
    font_id = 3 + 2 * n
    for i, text in enumerate(pages):
        content = f"BT /F1 {size} Tf 20 {150 if size < 40 else 100} Td ({text}) Tj ET".encode()
        objs.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 300] "
            f"/Contents {4 + 2 * i} 0 R /Resources << /Font << /F1 {font_id} 0 R >> >> >>"
            .encode())
        objs.append(b"<< /Length %d >>\nstream\n" % len(content) + content
                    + b"\nendstream")
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, o in enumerate(objs, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + o + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += (b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n"
            % (len(objs) + 1, xref))
    return bytes(out)


LONG1 = "The quick brown fox jumps over the lazy dog near the river bank"
LONG2 = "Second page carries different words about quarterly budget review"


class PdfExtractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def write(self, name: str, data: bytes) -> str:
        p = os.path.join(self.tmp.name, name)
        with open(p, "wb") as f:
            f.write(data)
        return p

    def test_two_page_text(self):
        p = self.write("a.pdf", make_pdf([LONG1, LONG2]))
        text = extract_text(p, "a.pdf", ocr_mode="off")
        self.assertEqual(text, f"{LONG1}\n\n{LONG2}")
        self.assertNotIn("\r", text)

    def test_thin_page_off_mode_keeps_text_of_other_pages(self):
        p = self.write("b.pdf", make_pdf([LONG1, "hi"]))
        text = extract_text(p, "b.pdf", ocr_mode="off")
        self.assertEqual(text, f"{LONG1}\n\nhi")

    def test_all_thin_off_raises_scanned_error(self):
        p = self.write("c.pdf", make_pdf([""]))
        with self.assertRaisesRegex(ValueError, "OCR is disabled"):
            extract_text(p, "c.pdf", ocr_mode="off")

    def test_all_thin_without_backend_raises_install_hint(self):
        p = self.write("c.pdf", make_pdf([""]))
        with mock.patch("app.ocr.available", return_value=False):
            with self.assertRaisesRegex(ValueError, "Tesseract"):
                extract_text(p, "c.pdf", ocr_mode="auto")

    def test_ocr_failure_keeps_embedded_text(self):
        p = self.write("d.pdf", make_pdf([LONG1, "hi"]))
        with mock.patch("app.ocr.available", return_value=True), \
                mock.patch("app.ocr.ocr_pdf_pages", side_effect=RuntimeError("x")):
            text = extract_text(p, "d.pdf", ocr_mode="auto")
        self.assertEqual(text, f"{LONG1}\n\nhi")

    def test_ocr_text_appended_to_thin_page_only(self):
        p = self.write("e.pdf", make_pdf([LONG1, "hi"]))
        with mock.patch("app.ocr.available", return_value=True), \
                mock.patch("app.ocr.ocr_pdf_pages",
                           return_value={1: "scanned words"}) as m:
            text = extract_text(p, "e.pdf", ocr_mode="auto")
        m.assert_called_once_with(p, [1])
        self.assertEqual(text, f"{LONG1}\n\nhi\n\nscanned words")

    def test_corrupt_pdf_raises_value_error(self):
        p = self.write("bad.pdf", b"%PDF-1.4\nthis is not a pdf at all")
        with self.assertRaisesRegex(ValueError, "Could not read PDF"):
            extract_text(p, "bad.pdf", ocr_mode="off")

    def test_encrypted_pdf_raises_value_error(self):
        doc = pdfium.PdfDocument(self.write("plain.pdf", make_pdf([LONG1])))
        self.assertEqual(len(doc), 1)
        doc.close()
        # pdfium cannot write encrypted files; a PDF whose /Encrypt dict is
        # present but unusable must still surface as a clean ValueError.
        data = make_pdf([LONG1]).replace(
            b"/Root 1 0 R", b"/Root 1 0 R /Encrypt << /Filter /Standard /V 1 "
            b"/R 2 /O (xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx) "
            b"/U (xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx) /P -4 >> "
            b"/ID [<00> <00>]")
        p = self.write("enc.pdf", data)
        with self.assertRaisesRegex(ValueError, "Could not read PDF"):
            extract_text(p, "enc.pdf", ocr_mode="off")


@unittest.skipUnless(ocr.available(), "tesseract not installed")
class OcrTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_backend_name(self):
        self.assertEqual(ocr.backend_name(), "tesseract")

    def test_ocr_pdf_pages(self):
        p = os.path.join(self.tmp.name, "big.pdf")
        with open(p, "wb") as f:
            f.write(make_pdf(["Invoice Total Payable", "Quarterly Budget"], size=48))
        out = ocr.ocr_pdf_pages(p, [0, 1, 7, -1])
        self.assertEqual(sorted(out), [0, 1])
        self.assertIn("invoice", out[0].lower())
        self.assertIn("payable", out[0].lower())
        self.assertIn("budget", out[1].lower())

    def test_extract_scanned_pdf_end_to_end(self):
        p = os.path.join(self.tmp.name, "scan.pdf")
        with open(p, "wb") as f:
            f.write(make_pdf(["Hello"], size=48))
        # "Hello" is a thin page, so auto mode OCRs it.
        text = extract_text(p, "scan.pdf", ocr_mode="auto")
        self.assertIn("Hello", text)

    def _pgm(self) -> bytes:
        doc = pdfium.PdfDocument(self._pdf())
        page = doc[0]
        bmp = page.render(scale=200 / 72, grayscale=True)
        data = ocr._pgm_bytes(bmp)
        bmp.close()
        page.close()
        doc.close()
        return data

    def _pdf(self) -> str:
        p = os.path.join(self.tmp.name, "src.pdf")
        with open(p, "wb") as f:
            f.write(make_pdf(["Invoice Total Payable"], size=48))
        return p

    def test_ocr_image_from_bytes_and_path(self):
        pgm = self._pgm()
        self.assertTrue(pgm.startswith(b"P5\n"))
        self.assertIn("payable", ocr.ocr_image(pgm).lower())
        path = os.path.join(self.tmp.name, "img.pgm")
        with open(path, "wb") as f:
            f.write(pgm)
        self.assertIn("invoice", ocr.ocr_image(path).lower())


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        ocr.reset_cache()
        self.addCleanup(ocr.reset_cache)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_missing_env_path_falls_through(self):
        missing = os.path.join(self.tmp.name, "nope")
        with mock.patch.dict(os.environ, {"PROCURE_TESSERACT": missing}), \
                mock.patch("shutil.which", return_value=None), \
                mock.patch.object(ocr, "_COMMON_PATHS", ()):
            self.assertEqual(ocr._find_binary(), "")
            self.assertFalse(ocr.available())
            self.assertEqual(ocr.backend_name(), "")
            with self.assertRaisesRegex(RuntimeError, "PROCURE_TESSERACT"):
                ocr.ocr_image(b"x")

    @unittest.skipIf(os.name == "nt", "shell script stub")
    def test_env_binary_is_used(self):
        fake = os.path.join(self.tmp.name, "fake-tesseract")
        with open(fake, "w") as f:
            f.write('#!/bin/sh\necho "  fake text  "\n')
        os.chmod(fake, os.stat(fake).st_mode | stat.S_IXUSR)
        with mock.patch.dict(os.environ, {"PROCURE_TESSERACT": fake}):
            self.assertEqual(ocr._find_binary(), fake)
            self.assertEqual(ocr.ocr_image(b"whatever"), "fake text")

    @unittest.skipIf(os.name == "nt", "shell script stub")
    def test_nonzero_exit_reports_last_stderr_line(self):
        fake = os.path.join(self.tmp.name, "bad-tesseract")
        with open(fake, "w") as f:
            f.write('#!/bin/sh\necho first >&2\necho boom >&2\nexit 3\n')
        os.chmod(fake, os.stat(fake).st_mode | stat.S_IXUSR)
        with mock.patch.dict(os.environ, {"PROCURE_TESSERACT": fake}):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                ocr.ocr_image(b"x")


if __name__ == "__main__":
    unittest.main()
