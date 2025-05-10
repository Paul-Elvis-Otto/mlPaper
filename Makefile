# Simple Makefile with 'help' as the default target

.PHONY: help build-data clean-data

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
