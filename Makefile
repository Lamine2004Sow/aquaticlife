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
	@echo "  make evolve    - run GA prototype"
	@echo "  make train     - run RL placeholder script"
	@echo "  make test      - run pytest (when tests available)"
	@echo "  make clean     - remove caches and runs"

venv:
	$(SYS_PY) -m venv $(VENV)

install: venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e .

evolve: install
	$(PYTHON) scripts/evolve.py

train: install
	$(PYTHON) scripts/train_rl.py

test: install
	$(PYTHON) -m pytest

clean:
	rm -rf runs .pytest_cache
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
