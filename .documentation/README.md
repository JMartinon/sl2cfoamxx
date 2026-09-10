<!-- SPDX-FileCopyrightText: 2026 Jeremy Martinon -->
<!-- SPDX-License-Identifier: GPL-3.0-or-later -->

# Documentation maintenance

The [published documentation](https://jmartinon.github.io/sl2cfoamxx/) explains
the theory and numerical algorithms. Its Markdown sources, theme assets, and
build configuration live together in this directory. Installation and API
guides remain in the repository's [documentation index](../docs/README.md).

From the repository root, build or preview the site with:

```sh
python3 -m venv .venv-docs
.venv-docs/bin/python -m pip install -r .documentation/requirements.txt
.venv-docs/bin/python -m mkdocs build --strict --config-file .documentation/mkdocs.yml
.venv-docs/bin/python -m mkdocs serve --config-file .documentation/mkdocs.yml
```

The generated site is written to `.documentation/site/` and is ignored by Git.
The [Documentation workflow](../.github/workflows/docs.yml) checks the site
and publishes it on GitHub Pages after changes to `main` when the repository
variable `PUBLISH_DOCS` is `true`. Pull requests only build the site.

The [figure sources and rendering notes](figures/README.md) explain how to
update the committed SVG diagrams. Ordinary documentation builds use those
SVG files and require no TeX installation.

Check release metadata and local Markdown links with:

```sh
python3 .github/check_repository.py
```
