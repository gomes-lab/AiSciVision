.DEFAULT_GOAL := all

# Define the directory to search in
SEARCH_STRING := "aiscivision"
SEARCH_DIR := .
EXCLUDE_SEARCH_DIR := {logs/,Data/,final_results/,__pycache__,.git,.ruff_cache}
EXCLUDE_SEARCH_FILE := {Makefile,.gitignore}

# Find SEARCH_STRING
# Usage: make find-text SEARCH_STRING="aiscivision"
find-text:
	@echo "Finding $(SEARCH_STRING) in $(SEARCH_DIR)"
	@grep -irn "$(SEARCH_STRING)" $(SEARCH_DIR) --exclude-dir=$(EXCLUDE_SEARCH_DIR) --exclude=$(EXCLUDE_SEARCH_FILE) || echo "No results found."

# Find TODOs
find-todos:
	@echo "Finding TODOs in $(SEARCH_DIR)"
	$(MAKE) find-text SEARCH_STRING="TODO"

# Define the Python source files
PYTHON_FILES := $(shell find . -type f -name "*.py")

# Default lint target (runs on all files)
lint:
	ruff check $(PYTHON_FILES)

# Lint target for a specific file
# Usage: make lint-file FILE=your_file.py
lint-file:
	@if [ -z "$(FILE)" ]; then \
		echo "Please specify a file using FILE=your_file.py"; \
		exit 1; \
	fi
	@if [ ! -f "$(FILE)" ]; then \
		echo "File $(FILE) does not exist"; \
		exit 1; \
	fi
	ruff check $(FILE)

# Fix lint issues in all files using Black
fix-lint:
	ruff check --fix $(PYTHON_FILES)
	ruff format $(PYTHON_FILES)

# Fix lint issues in a specific file using Black
# Usage: make fix-lint-file FILE=your_file.py
fix-lint-file:
	@if [ -z "$(FILE)" ]; then \
		echo "Please specify a file using FILE=your_file.py"; \
		exit 1; \
	fi
	@if [ ! -f "$(FILE)" ]; then \
		echo "File $(FILE) does not exist"; \
		exit 1; \
	fi
	ruff check --fix $(FILE)
	ruff format $(FILE)

# Add this to your default or all target if you want it to run automatically
all: lint

# Phony targets
.PHONY: lint lint-file fix-lint fix-lint-file all

default: lint
