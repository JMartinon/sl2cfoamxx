<!-- SPDX-FileCopyrightText: 2026 Jeremy Martinon -->
<!-- SPDX-License-Identifier: GPL-3.0-or-later -->

# Editable theory figures

These TikZ files adapt Jeremy Martinon's supplied `Vertex_HowTo.tex` and
`TNTex.tex`. The vertex drawing follows the graphical presentation discussed
in [Donà and Frisoni, *How-To compute EPRL spin foam amplitudes*](https://arxiv.org/abs/2202.04360).
The tensor-network drawing retains all original connections and uses the
weighted tensor names of the theory chapter: $S_A,\ldots,S_E$ and
$B^{(2)},\ldots,B^{(5)}$.

After editing a figure, regenerate the committed SVG assets from the repository
root with `python3 .documentation/render_figures.py`. This requires `pdflatex`
(including TikZ, PGFPlots, and the standalone class) and `pdf2svg`. The script
uses a temporary build directory, disables TeX shell escape, outlines the
mathematical glyphs, and adds accessible descriptions. The SVG canvas is
transparent; an embedded light/dark palette keeps the network lines and external
labels legible. Tensor nodes retain their opaque pastel green/blue fills in both
themes, with fixed dark labels inside them. TikZ color mixtures such as
`green!15` mix with white; they do not make the fill transparent.
The website sets `color-scheme` from the selected theme, which also controls
the SVG's `prefers-color-scheme` media query. This applies to the hand-authored
`numerical-organization.svg` overview as well. Print uses the light palette.

The Pages workflow uses the committed SVG files and does not need a TeX
installation. Review the rendered figures after every TikZ change. The
original files in the author's research directory are not modified.
