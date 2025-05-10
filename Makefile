# Simple Makefile with 'help' as the default target
#
# Self-documentation convention:
# Add '## Description' to any target line you want to show in 'make help'.
# Example:
# my-target: ## This does something amazing
# ------------------------------------------------------------------------------

# Define default goal to be 'help'
.DEFAULT_GOAL := help

# Phony targets: these are not files
.PHONY: help build build-data clean clean-data clean-temp build-correlation build-explaintop model-compare

# Default target / Help
help: ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Available targets:"
	@awk 'BEGIN {FS = ":.*?##"; OFS = "\t"} /^[a-zA-Z0-9_-]+:.*?##/ { if ($$1 != "help" && $$1 != ".PHONY") printf "  %-20s %s\n", $$1, $$2 }' $(MAKEFILE_LIST) | sort
	@echo ""
	@echo "Convenience targets:"
	@echo "  build            Alias for 'build-data'"
	@echo "  clean            Alias for 'clean-data' and 'clean-temp'"

# --- Data Handling ---
build-data: ## Build data using setup.R
	@echo "Building data..."
	Rscript setup/setup.R
	@echo "Created data."

clean-data: ## Clean and rebuild the ./data/ directory
	@echo "Cleaning out data dir..."
	rm -rf ./data/
	@echo "Rebuilding data dir..."
	mkdir ./data/
	@echo "Data dir cleaned and rebuilt."

clean-temp: ## Clean and rebuild the ./temp/ directory
	@echo "Cleaning out the temp dir..."
	rm -rf ./temp/
	@echo "Rebuilding temp dir..."
	mkdir ./temp/
	@echo "Temp dir cleaned and rebuilt."

# --- Analysis & Modeling ---
build-correlation: ## Build correlation matrix for v2x_corr and all other variables
	@echo "Building the correlation between v2x_corr and all others..."
	uv run ./src/v2x_corr_to_all.py
	@echo "Correlation built."

build-explaintop: ## Get definitions for top 10 correlated variables
	@echo "Getting definitions for the top 10 correlated variables..."
	Rscript ./src/get_top_10_definitions.R
	@echo "Definitions retrieved."

model-compare: ## Run model comparison script
	@echo "Running Model Compare..."
	uv run ./models/model_comparison.py
	@echo "Model comparison finished."

# --- Convenience Targets ---
build: build-data ## Compile the project (builds data)

clean: clean-data clean-temp ## Remove build artifacts and temporary files
	@echo "All clean targets executed."
