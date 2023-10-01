VENV           = .venv
VENV_PYTHON    = $(VENV)/bin/python
SYSTEM_PYTHON  = $(or $(shell which python3), $(shell which python))
PYTHON         = $(or $(wildcard $(VENV_PYTHON)), $(SYSTEM_PYTHON))

.DEFAULT_GOAL := build

$(VENV_PYTHON):
	rm -rf $(VENV)
	$(SYSTEM_PYTHON) -m venv $(VENV)

venv: $(VENV_PYTHON)


deps:
	$(PYTHON) -m pip install -r requirements-dev.txt

.PHONY: venv deps

test:
	$(PYTHON) -m pytest --benchmark-skip tests 

.PHONY: test

build:
	$(PYTHON) -m pip install --verbose --no-build-isolation --editable .

install:
	$(SYSTEM_PYTHON) -m pip install .

.PHONY: build install

docs-deps:
	$(PYTHON) -m pip install -r docs/requirements.txt

docs:
	$(PYTHON) -m sphinx docs build/html

.PHONY: docs docs-deps
