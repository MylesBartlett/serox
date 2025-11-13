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
# run the ruff formatter on all .py files in "check" mode
uv tool run ruff format --check
# run the ruff linter on all .py files
uv tool run ruff check
# lint python docstrings with pydoclint
uv run pydoclint serox/

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
