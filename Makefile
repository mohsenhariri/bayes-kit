
SRC:=scorio/
JULIA_PROJECT:=julia/Scorio.jl

.PHONY: format format-check lint clean build install test pkg-check pkg-publish-test pkg-publish sync-version release-py release-jl jl-install jl-test jl-test-slow py-docs-build py-docs-clean py-docs-serve jl-docs-build jl-docs-clean jl-docs-serve landing-serve
.PHONY: test-eval-py test-rank-py test-rank-py-slow test-eval-jl test-rank-jl test-rank-jl-slow

format-check:
	ruff format --check $(SRC)
	ruff check $(SRC)

format:
	ruff format $(SRC)
	ruff check --fix $(SRC)

lint:
	ruff check $(SRC)
	mypy $(SRC)

sync-version:
	python scripts/sync_version.py

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	pip install --upgrade build
	python -m build

install:
	pip install -e ".[dev]"

test:
	pytest tests/

test-eval-py:
	pytest tests/eval

test-rank-py:
	pytest tests/rank -m "not slow"

test-rank-py-slow:
	pytest tests/rank -m slow

pkg-check: build
	python -m pip install --upgrade twine
	twine check dist/*

pkg-publish-test: pkg-check
	@echo "Publishing to TestPyPI..."
	twine upload -r testpypi dist/* --verbose

pkg-publish: pkg-check
	twine upload dist/* --verbose

release-py:
	./scripts/release_github.sh py

release-jl:
	./scripts/release_github.sh jl

jl-install:
	julia --project=$(JULIA_PROJECT) -e 'using Pkg; Pkg.instantiate()'
	julia --project=$(JULIA_PROJECT)/docs -e 'using Pkg; Pkg.develop(path="$(JULIA_PROJECT)"); Pkg.instantiate()'

jl-test:
	julia --project=$(JULIA_PROJECT) -e 'using Pkg; Pkg.test()'

jl-test-slow:
	SCORIO_JL_RUN_SLOW=1 julia --project=$(JULIA_PROJECT) -e 'using Pkg; Pkg.test()'

test-eval-jl:
	julia --project=$(JULIA_PROJECT) -e 'using Scorio; include("$(JULIA_PROJECT)/test/eval/test_eval_apis.jl")'

test-rank-jl:
	julia --project=$(JULIA_PROJECT) -e 'using Scorio; include("$(JULIA_PROJECT)/test/rank/runtests_rank.jl")'

test-rank-jl-slow:
	SCORIO_JL_RUN_SLOW=1 julia --project=$(JULIA_PROJECT) -e 'using Scorio; include("$(JULIA_PROJECT)/test/rank/runtests_rank.jl")'

# Documentation

## Read the Docs
py-docs-build:
	cd docs && make html

py-docs-clean:
	cd docs && make clean

py-docs-serve:
	python -m http.server --directory docs/_build/html 4000

## Julia Docs
jl-docs-build:
	julia --project=$(JULIA_PROJECT)/docs $(JULIA_PROJECT)/docs/make.jl

jl-docs-clean:
	rm -rf $(JULIA_PROJECT)/docs/build

jl-docs-serve:
	python -m http.server --directory $(JULIA_PROJECT)/docs/build 4001

landing-serve:
	python -m http.server --directory docs-landing 4002
