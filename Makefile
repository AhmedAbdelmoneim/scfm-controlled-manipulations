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

## Paired structure metrics (embedding vs reference) into results_dir/evaluation
.PHONY: evaluate
evaluate:
	@scripts/run_evaluate.sh $(CONFIG)

## Run generated per-dataset evaluations, continuing after individual failures
.PHONY: evaluate-generated
evaluate-generated:
	@scripts/run_generated_evaluations.sh

## Reference-only scIB bio/batch metrics into results_dir/evaluation/{model}_scib_metrics.csv
.PHONY: evaluate-scib
evaluate-scib:
	@scripts/run_evaluate_scib.sh $(CONFIG)

## Reference-only trajectory metrics into results_dir/evaluation/{model}_trajectory_metrics.csv
.PHONY: evaluate-trajectory
evaluate-trajectory:
	@scripts/run_evaluate_trajectory.sh $(CONFIG)

## Reference-only enabled benchmarks across all datasets (foreground; pass ARGS="--dry-run")
.PHONY: evaluate-reference-benchmarks
evaluate-reference-benchmarks:
	@scripts/run_reference_benchmarks.sh --config $(CONFIG) $(ARGS)

## Profile evaluate on a synthetic fixture (see scripts/benchmark_eval.py)
.PHONY: benchmark-eval
benchmark-eval:
	uv run python scripts/benchmark_eval.py --config configs/experiments/atlases.yaml

## Validate manipulation h5ads for embedding (raw counts in X, gene/Ensembl metadata)
.PHONY: validate-embed-inputs
validate-embed-inputs:
	uv run python scripts/validate_embed_inputs.py \
		--dir "$(or $(MANIPULATIONS_DIR),$(CURDIR)/results/manipulations)" \
		$(if $(CONFIG),--config $(CONFIG),) \
		$(if $(FM_REPO),--fm-repo $(FM_REPO),) \
		$(foreach spec,$(GENE_LIST),--gene-list $(spec))

## Run unit tests
.PHONY: test
test:
	PYTHONPATH=metrics_dashboard uv run python -m unittest discover -s tests -v

## Fast synthetic smoke-run for evaluate pipeline
.PHONY: smoke-eval
smoke-eval:
	uv run python scripts/benchmark_eval.py --config configs/experiments/atlases.yaml --n-cells 200 --n-genes 400 --emb-dim 32 --max-interventions 1

## Export minimal Parquet dashboard bundles from SCEval dataset tree(s)
.PHONY: export-dashboard-bundle
export-dashboard-bundle:
	@test -n "$(SOURCE)" || (echo "Usage: make export-dashboard-bundle SOURCE=/path/to/sceval[/dataset] [OUTPUT=data/dashboard_bundles]" && exit 1)
	PYTHONPATH=metrics_dashboard uv run python scripts/export_dashboard_bundle.py \
		--source "$(SOURCE)" --output "$(or $(OUTPUT),$(CURDIR)/data/dashboard_bundles)"

## Streamlit metrics dashboard (metrics_dashboard/)
.PHONY: dashboard
dashboard:
	uv run --directory metrics_dashboard streamlit run Home.py



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
