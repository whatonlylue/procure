"""procure agent skill: one canonical file, shipped to every harness.

The skill lives at ``skills/procure/SKILL.md`` (repo root, Agent Skills
format). That single file is:

- served live as the ``procure://guide`` MCP resource,
- picked up by Claude Code via the plugin manifests at the repo root,
- copied into harness skill dirs by ``install-skills`` (CLI + desktop UI).

Runtime lookup covers the two supported run modes: a source checkout
(path relative to this file) and the frozen desktop sidecar (PyInstaller
``datas`` land under ``sys._MEIPASS``). Plain wheel installs carry no
skills tree, so the resolver degrades to a clear error there.
"""
from __future__ import annotations

import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

SKILL_NAME = "procure"
SKILL_FILE = "SKILL.md"


def skill_source_dir() -> Path:
    """Locate the bundled ``skills/procure/`` directory.

    Raises FileNotFoundError when neither the frozen bundle nor a source
    checkout provides it.
    """
    candidates: list[Path] = []
    meipass = getattr(sys, "_MEIPASS", None)
    if getattr(sys, "frozen", False) and meipass:
        candidates.append(Path(meipass) / "skills" / SKILL_NAME)
    candidates.append(
        Path(__file__).resolve().parent.parent / "skills" / SKILL_NAME)
    for c in candidates:
        if (c / SKILL_FILE).is_file():
            return c
    raise FileNotFoundError(
        "Bundled procure skill not found (looked in "
        + ", ".join(str(c) for c in candidates) + ")")


def skill_text() -> str:
    """Full SKILL.md text, frontmatter included."""
    return (skill_source_dir() / SKILL_FILE).read_text(encoding="utf-8")


def frontmatter_field(text: str, field: str) -> str:
    """Read one ``field: value`` from SKILL.md frontmatter.

    Handles top-level keys (``name``) and one nesting level
    (``metadata:`` -> ``version``). Returns "" when absent.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return ""
    in_meta = False
    for line in lines[1:]:
        if line.strip() == "---":
            break
        if not line.startswith((" ", "\t")):
            in_meta = line.startswith("metadata:")
            m = re.match(rf"^{re.escape(field)}:\s*['\"]?([^'\"\s#]+)",
                         line)
            if m:
                return m.group(1)
        elif in_meta:
            m = re.match(rf"^\s+{re.escape(field)}:\s*['\"]?([^'\"\s#]+)",
                         line)
            if m:
                return m.group(1)
    return ""


def skill_version(text: str | None = None) -> str:
    """Bundled skill version from frontmatter ("" when unknown)."""
    if text is None:
        try:
            text = skill_text()
        except FileNotFoundError:
            return ""
    return frontmatter_field(text, "version")


@dataclass(frozen=True)
class SkillTarget:
    """One harness skill directory (user-level, cross-harness format)."""

    id: str
    label: str
    home_subdir: str  # relative to ~, e.g. ".claude/skills"

    def path(self) -> Path:
        return Path.home() / self.home_subdir / SKILL_NAME

    def detected(self) -> bool:
        """Best-effort signal that the harness is in use."""
        if self.id == "claude":
            return ((Path.home() / ".claude").is_dir()
                    or shutil.which("Muse") is not None)
        return (Path.home() / self.home_subdir).is_dir()


TARGETS: tuple[SkillTarget, ...] = (
    SkillTarget("claude", "Claude Code", ".claude/skills"),
    SkillTarget("agents", "Universal (.agents)", ".agents/skills"),
)


def get_target(target_id: str) -> SkillTarget:
    for t in TARGETS:
        if t.id == target_id:
            return t
    known = ", ".join(t.id for t in TARGETS)
    raise ValueError(f"Unknown skill target: {target_id!r} (want {known})")


def _resolve_targets(target_ids: list[str] | None) -> list[SkillTarget]:
    if not target_ids:
        return list(TARGETS)
    seen: list[SkillTarget] = []
    for tid in target_ids:
        t = get_target(tid.strip().lower())
        if t not in seen:
            seen.append(t)
    return seen


def target_state(target: SkillTarget, bundled_version: str = "") -> dict:
    """Installed/available state of one target for status displays."""
    installed_file = target.path() / SKILL_FILE
    installed_version = ""
    if installed_file.is_file():
        try:
            installed_version = frontmatter_field(
                installed_file.read_text(encoding="utf-8"), "version")
        except OSError:
            installed_version = ""
    installed = installed_file.is_file()
    return {
        "id": target.id,
        "label": target.label,
        "path": str(target.path()),
        "detected": target.detected(),
        "installed": installed,
        "installed_version": installed_version,
        "bundled_version": bundled_version,
        "outdated": installed and installed_version != bundled_version,
    }


def status() -> dict:
    """Skill bundle + per-target install state (CLI and UI share this)."""
    try:
        src = skill_source_dir()
        bundled = skill_version()
        found = True
    except FileNotFoundError:
        src, bundled, found = None, "", False
    return {
        "skill": SKILL_NAME,
        "source_found": found,
        "source_path": str(src) if src else "",
        "version": bundled,
        "targets": [target_state(t, bundled) for t in TARGETS],
    }


def install(target_ids: list[str] | None = None,
            force: bool = False) -> dict:
    """Copy the bundled skill into harness skill dirs.

    Installs to every known target unless ``target_ids`` narrows it.
    Existing installs are skipped unless ``force`` is set; installing is
    what picks up a new bundled version. Per-target errors are recorded
    without aborting the remaining targets.
    """
    src = skill_source_dir()  # raises FileNotFoundError when unavailable
    results: list[dict] = []
    for target in _resolve_targets(target_ids):
        dest = target.path()
        try:
            if (dest / SKILL_FILE).is_file() and not force:
                results.append({"id": target.id, "path": str(dest),
                                "action": "skipped", "detail": "already installed"})
                continue
            action = "updated" if dest.is_dir() else "installed"
            if dest.is_dir():
                shutil.rmtree(dest)
            shutil.copytree(src, dest)
            results.append({"id": target.id, "path": str(dest),
                            "action": action, "detail": ""})
        except OSError as e:
            results.append({"id": target.id, "path": str(dest),
                            "action": "error", "detail": str(e)})
    return {"results": results}
