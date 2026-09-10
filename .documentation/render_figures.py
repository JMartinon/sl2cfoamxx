#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Jeremy Martinon
# SPDX-License-Identifier: GPL-3.0-or-later

"""Render the two author-supplied TikZ diagrams as self-contained SVG assets."""

from pathlib import Path
import shutil
import subprocess
import tempfile
from xml.sax.saxutils import escape


DOCUMENTATION = Path(__file__).resolve().parent
ROOT = DOCUMENTATION.parent
FIGURES = {
    "vertex-booster-decomposition": (
        "Booster decomposition of the Lorentzian EPRL vertex",
        "The gauge-fixed tetrahedron 1 has intertwiner i1 and no booster. "
        "Four booster diagrams join boundary spins j_ab to the lifted spins "
        "ell_ab and internal intertwiners k2 through k5 of the SU(2) network.",
    ),
    "vertex-tensor-network": (
        "Tensor network for the full rank-five Lorentzian EPRL vertex",
        "Five green weighted 6j tensors S_A through S_E and four blue booster "
        "tensors B^(2) through B^(5) are connected by labeled index junctions. "
        "The five boundary intertwiners remain open; all other indices are summed. "
        "Crossings without a labeled junction do not connect.",
    ),
}

# Preserve the TikZ geometry while making its ink respond to the embedding
# page's color-scheme. The SVG canvas remains transparent in either theme.
PALETTE = {
    "ink": ("rgb(0%, 0%, 0%)", "#d5dee8"),
    "inverse": ("rgb(100%, 100%, 100%)", "#20232d"),
    "red": ("rgb(79.998779%, 0%, 0%)", "#ff9696"),
    "blue": ("rgb(25%, 25%, 100%)", "#a2b7ff"),
    "sixj": ("rgb(84.999084%, 100%, 84.999084%)", "#274b3e"),
    "booster": ("rgb(84.999084%, 84.999084%, 100%)", "#303e68"),
}


def themed_svg(document):
    for name, (light, _) in PALETTE.items():
        for attribute in ("fill", "stroke"):
            document = document.replace(
                f'{attribute}="{light}"', f'{attribute}="var(--figure-{name})"'
            )
    light = " ".join(f"--figure-{name}: {colors[0]};" for name, colors in PALETTE.items())
    dark = " ".join(f"--figure-{name}: {colors[1]};" for name, colors in PALETTE.items())
    style = (
        f"\n<style>\n:root {{ {light} }}\n"
        f"@media (prefers-color-scheme: dark) {{ :root {{ {dark} }} }}\n"
        f"@media print {{ :root {{ {light} }} }}\n</style>"
    )
    return document, style


def main():
    for executable in ("pdflatex", "pdf2svg"):
        if not shutil.which(executable):
            raise SystemExit(f"Required figure-rendering tool not found: {executable}")

    assets = DOCUMENTATION / "theory/assets"
    with tempfile.TemporaryDirectory(prefix="sl2cfoam-figures-") as directory:
        build = Path(directory)
        for name, (title, description) in FIGURES.items():
            source = DOCUMENTATION / "figures" / f"{name}.tex"
            command = [
                "pdflatex", "-no-shell-escape", "-interaction=nonstopmode",
                "-halt-on-error", f"-output-directory={build}", str(source),
            ]
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            if result.returncode:
                raise SystemExit(result.stdout + result.stderr)
            if "Missing character:" in result.stdout:
                raise SystemExit(f"Missing glyph in {source.name}:\n{result.stdout}")

            svg = build / f"{name}.svg"
            subprocess.run(["pdf2svg", str(build / f"{name}.pdf"), str(svg)], check=True)
            document, style = themed_svg(svg.read_text())
            start = document.index("<svg ")
            end = document.index(">", start)
            document = (
                document[:end]
                + ' role="img" aria-labelledby="figure-title figure-description"'
                + document[end:end + 1]
                + '\n<!-- SPDX-FileCopyrightText: 2026 Jeremy Martinon -->'
                + '\n<!-- SPDX-License-Identifier: GPL-3.0-or-later -->'
                + f'\n<title id="figure-title">{escape(title)}</title>'
                + f'\n<desc id="figure-description">{escape(description)}</desc>'
                + style
                + document[end + 1:]
            )
            output = assets / f"{name}.svg"
            output.write_text(document)
            print(f"Rendered {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
