"""Fail-closed, text-only release inventory and common-secret scan."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IGNORED = {".git", "output", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache"}
ALLOWED_SUFFIXES = {".py", ".md", ".toml", ".json", ".csv"}
ALLOWED_DOTFILES = {".gitignore", ".gitattributes", ".python-version"}
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\b(?:ghp_|github_pat_)[A-Za-z0-9_]{20,}"),
    re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{24,}"),
)


def check() -> list[str]:
    paths = []
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if any(part in IGNORED for part in relative.parts) or path.is_dir():
            continue
        if path.name not in ALLOWED_DOTFILES and path.suffix not in ALLOWED_SUFFIXES:
            raise ValueError(f"Non-text or unreviewed asset: {relative}")
        if path.stat().st_size > 500_000:
            raise ValueError(f"Oversized text asset: {relative}")
        content = path.read_text(encoding="utf-8")
        if any(pattern.search(content) for pattern in SECRET_PATTERNS):
            raise ValueError(f"Credential-shaped content: {relative}")
        paths.append(relative.as_posix())
    return paths


if __name__ == "__main__":
    files = check()
    print(f"PASS: {len(files)} reviewed text files; no binary or credential-shaped assets")
