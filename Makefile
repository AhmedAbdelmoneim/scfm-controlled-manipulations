#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = scfm-controlled-manipulations
PYTHON_VERSION = 3.11
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	uv sync
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	uv run ruff format --check .
	uv run ruff check .

## Format source code with ruff
.PHONY: format
format:
	uv run ruff check --fix .
	uv run ruff format .





## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> New uv virtual environment created. Activate with:"
	@echo ">>> Windows: .\\\\.venv\\\\Scripts\\\\activate"
	@echo ">>> Unix/macOS: source ./.venv/bin/activate"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

CONFIG ?= configs/default.yaml

## Run interventions on input_h5ad (writes results/manipulations/*.h5ad)
.PHONY: manipulate
manipulate:
	uv run python -m scfm_controlled_manipulations.pipeline manipulate --config $(CONFIG)

## Paired structure metrics (raw + embedding vs reference) into results_dir/evaluation
.PHONY: evaluate
evaluate:
	PYTHONUNBUFFERED=1 uv run python -m scfm_controlled_manipulations.pipeline evaluate --config $(CONFIG)



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
