# Φ — one-command entrypoints. Run `make help` to see everything.
# These are thin wrappers; the real work is LeRobot CLI commands documented in docs/.
.PHONY: help setup lint test docs submodule

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n",$$1,$$2}'

submodule:  ## Pull the pinned LeRobot engine
	git submodule update --init --recursive

setup:  ## Create the conda env (override: make setup ENV=cuda)
	conda env create -f env/environment.$(or $(ENV),mac).yml

lint:  ## ruff + mypy
	ruff check src tests && mypy src

test:  ## unit + smoke tests
	pytest -q tests

docs:  ## Serve the docs site locally (mkdocs)
	mkdocs serve

# --- Pipeline shortcuts (SO-101). Fill ROBOT_PORT / LEADER_PORT / HF_USER in your shell. ---
.PHONY: calibrate teleop record train eval
calibrate: ; @echo "See docs/robots/so-arm101/02-setup.md (lerobot-calibrate)"
teleop:    ; @echo "See docs/robots/so-arm101/02-setup.md (lerobot-teleoperate)"
record:    ; @echo "See docs/robots/so-arm101/03-teleop-and-data.md (lerobot-record)"
train:     ; @echo "See docs/training/README.md (lerobot-train)"
eval:      ; @echo "See docs/evaluation/README.md (lerobot-rollout)"
