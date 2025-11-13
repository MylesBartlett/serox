#!/bin/bash
# Configure to exit on error
set -e

# Increment version based on argument (major | minor | patch)
uv run bump2version $1

# Build the package
uv build

# Publish to PyPI
uv publish
