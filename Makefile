UV ?= uv
BUILD_EXT_ARGS ?=
TESTS ?= tests
PYTEST_ARGS ?=
DEPS_STAMP = .venv/.wildboar-deps
DEPS_INPUTS = pyproject.toml $(wildcard uv.lock)

.DEFAULT_GOAL := build

# Sync once, then again only when dependency metadata changes.
$(DEPS_STAMP): $(DEPS_INPUTS)
	$(UV) sync
	touch $@

deps: $(DEPS_STAMP)

# Reuse compilation artifacts between dependency syncs.
build: deps
	$(UV) run --no-sync python setup.py build_ext --inplace $(BUILD_EXT_ARGS)

rebuild: deps
	$(UV) run --no-sync python setup.py build_ext --inplace --force $(BUILD_EXT_ARGS)

test: deps
	$(UV) run --no-sync python -m pytest --benchmark-skip $(TESTS) $(PYTEST_ARGS)

# Install a regular (non-editable) build into the project environment.
install:
	$(UV) pip install .

docs-deps: deps
	$(UV) pip install -r docs/requirements.txt

docs:
	$(UV) run --no-sync python -m sphinx docs build/html

.PHONY: deps build rebuild test install docs-deps docs
