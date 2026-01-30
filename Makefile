# Simple workflow helpers for the aquaticlife project

VENV ?= .venv
# System interpreter used to create the venv (python3 is present even when "python" is absent)
SYS_PY ?= python3
# Interpreter inside the venv
PYTHON := $(VENV)/bin/python

.PHONY: help venv install evolve train test clean

help:
	@echo "Targets:"
	@echo "  make venv      - create local virtualenv ($(VENV))"
	@echo "  make install   - install package editable + deps"
	@echo "  make install-rl  - install with RL extra (PyTorch)"
	@echo "  make install-viz - install with viz extra (pygame)"
	@echo "  make install-dev - install with dev extras (pytest...)"
	@echo "  make evolve    - run GA prototype"
	@echo "  make train     - run RL placeholder script"
	@echo "  make view      - run pygame viewer (needs pygame)"
	@echo "  make test      - run pytest (when tests available)"
	@echo "  make clean     - remove caches and runs"

venv:
	$(SYS_PY) -m venv $(VENV)

install: venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e .

install-rl: venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[rl]"

install-viz: venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[viz]"

install-dev: venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

evolve: install
	$(PYTHON) scripts/evolve.py

train: install-rl
	$(PYTHON) scripts/train_rl.py

view: install-viz
	$(PYTHON) scripts/viewer.py

test: install-dev
	$(PYTHON) -m pytest

clean:
	rm -rf runs .pytest_cache
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
