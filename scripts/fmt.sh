#!/usr/bin/sh

# install the formatters not managed by rye
. ./scripts/install-linters-formatters.sh

# md
echo "[dprint] Formatting .md files..."
dprint fmt

# toml
echo "[taplo] Formatting .toml files..."
taplo format

# yaml
echo "[yamlfmt] Formatting .y[a]ml..."
yamlfmt

# python
echo "[ruff] Formatting .py files..."
rye run fmt

echo "✨ Done formatting ✨"
