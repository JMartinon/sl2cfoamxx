#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Jeremy Martinon
# SPDX-License-Identifier: GPL-3.0-or-later

"""Check release-version consistency and local documentation links; no packages needed."""

from __future__ import annotations

import argparse
import html
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote, urlsplit


def prose(text: str) -> str:
    """Exclude fenced examples when looking for headings and links."""
    lines = []
    fence = None
    for line in text.splitlines():
        match = re.match(r"^\s*(`{3,}|~{3,})", line)
        if match:
            marker = match.group(1)
            if fence is None:
                fence = marker
            elif marker[0] == fence[0] and len(marker) >= len(fence):
                fence = None
            lines.append("")
        else:
            lines.append(line if fence is None else "")
    return "\n".join(lines)


def anchors(text: str) -> set[str]:
    """GitHub-style ATX heading slugs, duplicates, and explicit HTML anchors."""
    text = prose(text)
    result = set(re.findall(r'\b(?:id|name)=["\']([^"\']+)["\']', text))
    counts: dict[str, int] = {}
    for heading in re.findall(r"^ {0,3}#{1,6}\s+(.+?)\s*#*\s*$", text, re.M):
        heading = re.sub(r"\[([^]]+)\]\([^)]*\)", r"\1", heading)
        heading = re.sub(r"<[^>]+>", "", heading)
        heading = html.unescape(heading).lower().replace("`", "")
        slug = re.sub(r"[^\w\- ]", "", heading).replace(" ", "-")
        number = counts.get(slug, 0)
        counts[slug] = number + 1
        result.add(f"{slug}-{number}" if number else slug)
    return result


def destinations(text: str):
    """Read ordinary Markdown links/references and HTML href/src attributes."""
    text = prose(text)
    # Literal inline examples such as `[label](path)` are not navigable links.
    text = re.sub(r"(`+).*?\1", "", text)
    definitions = {}
    for label, target in re.findall(r"^ {0,3}\[([^]]+)\]:\s*(\S+)", text, re.M):
        definitions[label.casefold()] = target.strip("<>")
    for match in re.finditer(r"!?\[[^]\n]*\]\((<[^>]*>|[^\s)]*)(?:\s+[^)]*)?\)", text):
        yield match.group(1).strip("<>")
    for label, reference in re.findall(r"!?\[([^]\n]+)\]\[([^]\n]*)\]", text):
        reference = (reference or label).casefold()
        if reference not in definitions:
            raise ValueError(f"undefined Markdown link reference [{reference}]")
        yield definitions[reference]
    for target in definitions.values():
        yield target
    yield from re.findall(r'\b(?:href|src)=["\']([^"\']+)["\']', text)


def document_paths(root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", "*.md"],
        cwd=root, capture_output=True, check=False,
    )
    if result.returncode == 0:
        return sorted({root / path for path in result.stdout.decode().split("\0") if path})
    # Source archives need not retain .git; restrict scanning to maintained docs.
    paths = list((root / "docs").rglob("*.md"))
    paths += list((root / ".documentation").rglob("*.md"))
    paths += [root / name for name in ("README.md", "src/README.md", "mathematica/README.md", "mathematica/protocol.md")]
    return sorted(set(path for path in paths if path.is_file()))


def check_versions(root: Path) -> str:
    patterns = {
        "CMakeLists.txt": r"project\(sl2cfoamxx\s+VERSION\s+(\d+\.\d+\.\d+)",
        "julia/SL2CfoamXX/Project.toml": r'^version\s*=\s*"(\d+\.\d+\.\d+)"',
        "CITATION.cff": r'^version:\s*["\']?(\d+\.\d+\.\d+)',
        "README.md": r"version\s*=\s*\{(\d+\.\d+\.\d+)\}",
    }
    versions = {}
    for name, pattern in patterns.items():
        match = re.search(pattern, (root / name).read_text(), re.M)
        if not match:
            raise ValueError(f"{name}: release version not found")
        versions[name] = match.group(1)
    if len(set(versions.values())) != 1:
        raise ValueError("inconsistent release versions: " + ", ".join(f"{name}={version}" for name, version in versions.items()))
    return versions["CMakeLists.txt"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    root = parser.parse_args().root.resolve()
    errors = []
    try:
        version = check_versions(root)
    except (OSError, ValueError) as error:
        errors.append(str(error))
        version = "invalid"
    checked = 0
    anchor_cache = {}
    documents = document_paths(root)
    for document in documents:
        if not document.is_file():  # Tracked deletions are checked through their incoming links.
            continue
        try:
            for destination in destinations(document.read_text()):
                url = urlsplit(html.unescape(destination))
                if url.scheme or url.netloc or not destination:
                    continue
                relative = unquote(url.path)
                target = (root / relative.lstrip("/") if relative.startswith("/")
                          else document.parent / relative) if relative else document
                target = target.resolve()
                if not target.is_relative_to(root):
                    errors.append(f"{document.relative_to(root)}: link leaves repository: {destination}")
                elif not target.exists():
                    errors.append(f"{document.relative_to(root)}: missing target: {destination}")
                elif url.fragment and target.suffix.lower() == ".md":
                    if target not in anchor_cache:
                        anchor_cache[target] = anchors(target.read_text())
                    if unquote(url.fragment) not in anchor_cache[target]:
                        errors.append(f"{document.relative_to(root)}: missing heading/anchor: {destination}")
                checked += 1
        except (OSError, ValueError) as error:
            errors.append(f"{document.relative_to(root)}: {error}")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"Release {version}: consistent metadata; {checked} local links checked in {len(documents)} Markdown files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
