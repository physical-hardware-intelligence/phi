# Φ — one-command entrypoints. Run `make help` to see everything.
# These are thin wrappers; the real work is LeRobot CLI commands documented in docs/.
.PHONY: help setup install policies doctor lint test docs

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n",$$1,$$2}'

setup:  ## Create the conda env (override: make setup ENV=cuda)
	conda env create -f env/environment.$(or $(ENV),mac).yml

install:  ## One-shot: env + phi + verify. Idempotent. (override: make install ENV=cuda)
	@if conda env list | grep -qE '^phi[[:space:]]'; then \
		echo '==> env phi exists, updating from env/environment.$(or $(ENV),mac).yml'; \
		conda env update -n phi -f env/environment.$(or $(ENV),mac).yml; \
	else \
		echo '==> creating env phi from env/environment.$(or $(ENV),mac).yml'; \
		conda env create -f env/environment.$(or $(ENV),mac).yml; \
	fi
	@echo '==> installing the phi package (editable)'
	conda run -n phi --no-capture-output pip install -e .
	@echo '==> verifying'
	@conda run -n phi python -c "import torch, lerobot, torchcodec, sys; \
print('python     ', sys.version.split()[0]); \
print('interpreter', sys.executable); \
print('lerobot    ', lerobot.__version__); \
print('torch      ', torch.__version__); \
print('torchcodec ', torchcodec.__version__); \
print('accelerator', 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'CPU ONLY'))"
	@echo ''
	@echo 'Done. Now run:  conda activate phi'

policies:  ## Add the per-policy extras (smolvla, pi, diffusion) to an existing env
	conda run -n phi --no-capture-output pip install \
		"lerobot[smolvla]==0.6.0" "lerobot[pi]==0.6.0" "lerobot[diffusion]==0.6.0"
	@echo 'pi0/pi0.5 also need: hf auth login, and the PaliGemma licence accepted.'

doctor:  ## Diagnose this machine: env, cameras, ports, calibration, HF auth
	@conda run -n phi --no-capture-output python -m phi.utils.doctor || true

lint:  ## ruff + mypy
	ruff check src tests && mypy src

test:  ## unit + smoke tests (includes the no-raw-cv2 guard)
	conda run -n phi --no-capture-output python -m pytest -q tests

docs:  ## Serve the docs site locally (mkdocs)
	mkdocs serve

# --- Pipeline shortcuts (SO-101). Fill ROBOT_PORT / LEADER_PORT / HF_USER in your shell. ---
.PHONY: calibrate teleop record train eval
calibrate: ; @echo "See docs/robots/so-arm101/02-setup.md (lerobot-calibrate)"
teleop:    ; @echo "See docs/robots/so-arm101/02-setup.md (lerobot-teleoperate)"
record:    ; @echo "See docs/robots/so-arm101/03-teleop-and-data.md (lerobot-record)"
train:     ; @echo "See docs/training/README.md (lerobot-train)"
eval:      ; @echo "See docs/evaluation/README.md (lerobot-rollout)"
