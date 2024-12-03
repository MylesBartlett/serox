#!/usr/bin/sh

# install the linters not managed by rye
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

# yaml
echo "[yamlfmt] Checking .y[a]ml..."
yamlfmt -lint

# spelling
## codespell
echo "[codespell] Checking spelling..."
uv run codespell

# dependencies
echo "[deptry] Checking for redundant dependencies..."
uv run deptry .

echo "✨ Done linting ✨"
