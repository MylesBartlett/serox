#!/usr/bin/env bash
# Check if gh is installed
if ! command -v gh &>/dev/null; then
	echo "'gh' not found but needed to run 'act'. See here for installation:
    https://github.com/cli/cli?tab=readme-ov-file#installation"
fi
# Check if gh extension `act` is installed
if ! gh extension list | grep -q "act"; then
	echo "'gh act' extension not found. Installing gh 'act' extension..."
	gh extension install nektos/gh-act
fi

# Run all workflows using gh act
echo "Running all workflows..."
gh act run
