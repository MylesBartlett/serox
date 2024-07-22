#!/usr/bin/sh
. ./scripts/install-linters-formatters.sh
# toml
echo "[taplo] Checking .toml..."
taplo format --check
# md, json, Dockerfile, ts, js
echo "[dprint] Checking .md files..."
dprint check
# python
echo "[ruff] Checking .py files..."
rye fmt --check
rye lint
# lint docstrings
rye run lint-doc
# spelling
## codespell
echo "[codespell] Checking spelling..."
rye run codespell

echo "✨ Done linting ✨"
