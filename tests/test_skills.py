"""Coverage for the procure agent skill bundle and its install paths.

Stdlib unittest only (no new harness). Run from the project root:

    .venv/bin/python -m unittest discover -s tests
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app import mcp_server, skills

REPO_ROOT = Path(__file__).resolve().parent.parent


class SkillSourceTest(unittest.TestCase):
    def test_resolves_repo_skill(self) -> None:
        src = skills.skill_source_dir()
        self.assertEqual(src, REPO_ROOT / "skills" / "procure")
        text = skills.skill_text()
        self.assertEqual(skills.frontmatter_field(text, "name"), "procure")
        self.assertTrue(skills.skill_version(text))

    def test_frontmatter_missing_file(self) -> None:
        self.assertEqual(skills.frontmatter_field("no frontmatter", "name"), "")
        self.assertEqual(
            skills.frontmatter_field("---\nname: x\n---\n", "version"), "")

    def test_guide_serves_skill_byte_identical(self) -> None:
        """procure://guide and the shipped skill are one file, no drift."""
        self.assertEqual(mcp_server.read_guide(), skills.skill_text())
        self.assertIn("procure_search", mcp_server.read_guide())

    def test_no_legacy_guide_file(self) -> None:
        self.assertFalse((REPO_ROOT / "app" / "mcp_guide.md").exists())


class SkillManifestTest(unittest.TestCase):
    def test_plugin_manifests_consistent(self) -> None:
        plugin = json.loads(
            (REPO_ROOT / ".claude-plugin" / "plugin.json").read_text())
        market = json.loads(
            (REPO_ROOT / ".claude-plugin" / "marketplace.json").read_text())
        self.assertEqual(plugin["name"], "procure")
        self.assertEqual(len(market["plugins"]), 1)
        entry = market["plugins"][0]
        self.assertEqual(entry["name"], plugin["name"])
        self.assertEqual(entry["version"], plugin["version"])
        self.assertEqual(entry["source"], "./")
        self.assertTrue(
            (REPO_ROOT / "skills" / plugin["name"] / "SKILL.md").is_file())

    def test_mcp_json_points_at_local_server(self) -> None:
        conf = json.loads((REPO_ROOT / ".mcp.json").read_text())
        srv = conf["mcpServers"]["procure"]
        self.assertIn("http", srv["type"])
        self.assertTrue(srv["url"].endswith("/mcp"))


class SkillInstallTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self._patch = mock.patch.dict(os.environ, {"HOME": self.tmp.name})
        self._patch.start()
        self.addCleanup(self._patch.stop)
        self.addCleanup(self.tmp.cleanup)

    def _installed_text(self, target_id: str) -> str:
        t = skills.get_target(target_id)
        return (t.path() / skills.SKILL_FILE).read_text(encoding="utf-8")

    def test_install_all_then_skip_then_force(self) -> None:
        first = skills.install()
        self.assertEqual(
            {r["action"] for r in first["results"]}, {"installed"})
        for t in skills.TARGETS:
            self.assertEqual(self._installed_text(t.id), skills.skill_text())

        second = skills.install()
        self.assertEqual(
            {r["action"] for r in second["results"]}, {"skipped"})

        third = skills.install(force=True)
        self.assertEqual(
            {r["action"] for r in third["results"]}, {"updated"})

    def test_install_subset_and_unknown_target(self) -> None:
        out = skills.install(["claude"])
        self.assertEqual([r["id"] for r in out["results"]], ["claude"])
        self.assertTrue(skills.get_target("claude").path().is_dir())
        self.assertFalse(skills.get_target("agents").path().exists())
        with self.assertRaises(ValueError):
            skills.install(["nope"])

    def test_status_reports_outdated(self) -> None:
        skills.install(["claude"])
        dest = skills.get_target("claude").path() / skills.SKILL_FILE
        dest.write_text("---\nname: procure\nmetadata:\n  version: 0.0.0\n---\nold\n",
                        encoding="utf-8")
        st = skills.status()
        self.assertTrue(st["source_found"])
        by_id = {t["id"]: t for t in st["targets"]}
        self.assertTrue(by_id["claude"]["installed"])
        self.assertTrue(by_id["claude"]["outdated"])
        self.assertFalse(by_id["agents"]["installed"])
        self.assertFalse(by_id["agents"]["outdated"])


if __name__ == "__main__":
    unittest.main()
