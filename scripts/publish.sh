#!/bin/bash
# Configure to exit on error
set -e

# Increment version based on argument (major | minor | patch)
rye run bump2version $1

# Build the package
rye build

# Publish to PyPI
rye publish
