# Simple Makefile with 'help' as the default target

.PHONY: help build-data clean-data build-correlation build-explaintop

# Default target
help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Available targets:"
	@echo "  build    Compile the project"
	@echo "  clean    Remove build artifacts"

build-data:
	@echo "Building data..."
	Rscript setup/setup.R
	@echo "created data"

build-correlation:
	@echo "build the correlation between v2x_corr and all other"
	uv run ./src/v2x_corr_to_all.py
	@echo "done"

build-explaintop:
	@echo "Get the definitions for the top 10 corr vars"
	Rscript ./src/get_top_10_definitions.R
	@echo "done"

clean-data:
	@echo "Cleaning out data dir..."
	rm -rf ./data/
	@echo "Rebuilding data dir..."
	mkdir ./data/
	@echo "Done"

clean-temp:
	@echo "Cleaning out the temp dir..."
	rm -rf ./temp/
	@echo "Rebuilding temp dir..."
	mkdir ./temp/
	@echo "Done"
