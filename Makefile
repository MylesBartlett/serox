.DEFAULT_GOAL := help

SHELL=/bin/bash
BASE ?= main

.PHONY: fmt
fmt:  ## Run formatters
	@. ./scripts/fmt.sh

.PHONY: lint
lint:  ## Run linters
	@. ./scripts/lint.sh

.PHONY: test
test:  ## Run tests
	@rye test

.PHONY: pre-commit
pre-commit: ## Run pre-commit hooks
	@rye run pre-commit run

.PHONY: workflow
workflow:
	@. ./scripts/run-workflows.sh

.PHONY: clean
clean:  ## Clean up caches and build artifacts
	@rm -rf .ruff_cache/
	@rm -rf .venv/
	@rm -rf .pytest_cache/

.PHONY: bump-patch
bump-patch: ## Bump patch version and publish
	@. ./scripts/publish.sh patch

.PHONY: bump-minor
bump-minor: ## Bump minor version and publish
	@. ./scripts/publish.sh minor

.PHONY: bump-major
bump-major: ## Bump major version and publish
	@. ./scripts/publish.sh major

.PHONY: help
help:  ## Display this help screen
	@echo -e "\033[1mAvailable commands:\033[0m"
	@grep -E '^[a-z.A-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' | sort
