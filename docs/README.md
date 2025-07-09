<!--
Copyright (C) 2024 Roberto Rossini <roberros@uio.no>

SPDX-License-Identifier: MIT
-->

# Documentation README

## How to build StripePy's documentation

The instructions in this README assume all commands are being run from the root of StripePy's repository.

```bash
venv/bin/pip install '.[all,docs]' -v

# Activate venv
. venv/bin/activate

# Clean old build files (optional)
make -C docs clean

make -C docs linkcheck html latexpdf
```

Open the HTML documentation:

```bash
# Linux
xdg-open docs/_build/html/index.html

# macOS
open docs/_build/html/index.html
```

Open the PDF documentation:

```bash
# Linux
xdg-open docs/_build/latex/stripepy.pdf

# macOS
open docs/_build/latex/stripepy.pdf
```

## How to automatically generate documentation for the CLI

```bash
venv/bin/python docs/generate_cli_reference.py \
  --stripepy venv/bin/stripepy |
  tee docs/cli_reference.rst
```
