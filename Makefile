tag=gym-duckiematrix
PYTHON ?= python3
TEST_WORKERS ?= 8
PYTEST_ARGS ?=
export PYTHONPATH := $(CURDIR)/src$(if $(PYTHONPATH),:$(PYTHONPATH))

regular_packages=gym_duckiematrix
test_packages=gym_duckiematrix_tests
cover_packages=$(test_packages),$(regular_packages)


CIRCLE_NODE_INDEX ?= 0
CIRCLE_NODE_TOTAL ?= 1

out=out
coverage_dir=$(out)/coverage
tr=$(out)/test-results
xunit_output=$(tr)/pytest-$(CIRCLE_NODE_INDEX)-xunit.xml


test_files=src/gym_duckiematrix_tests
coverage=--cov=$(regular_packages) --cov-branch --cov-report=term-missing --cov-report=html:$(coverage_dir)
pytest=$(PYTHON) -W default -m pytest $(test_files) -v $(coverage) --junitxml=$(xunit_output) $(PYTEST_ARGS)



all:
	@echo "You can try:"
	@echo
	@echo "  make build run"
	@echo "  make docs "
	@echo "  make test coverage-combine coverage-report"
	@echo "  "
	@echo "  make -C notebooks clean all"



black:
	black -l 79 --target-version py312 src examples

clean:
	$(PYTHON) -m coverage erase
	rm -rf $(out) $(coverage_dir) $(tr)

test: clean
	mkdir -p $(tr)
	DISABLE_CONTRACTS=1 MPLBACKEND=Agg $(pytest)


test-parallel: clean
	mkdir -p  $(tr)
	DISABLE_CONTRACTS=1 MPLBACKEND=Agg $(pytest) -n $(TEST_WORKERS)


test-parallel-circle: test-parallel


coverage-combine:
	$(PYTHON) -m coverage report



build:
	docker build -t $(tag) .

build-no-cache:
	docker build --no-cache -t $(tag) .


test-docker: build
	docker run -it $(tag) make test


run:
	mkdir -p out-docker
	docker run -it -v $(PWD)/out-docker:/out $(tag) dt-pc-demo

run-with-mounted-src:
	mkdir -p out-docker
	docker run -it -v $(PWD)/src:/duckietown_pondcleaner/src:ro -v $(PWD)/out-docker:/out $(tag)


coverage-report:
	$(PYTHON) -m coverage html -d $(coverage_dir)

docs:
	sphinx-build src $(out)/docs


# Release code

bump-upload:
	$(MAKE) bump
	$(MAKE) upload

bump:
	bumpversion patch

upload:
	git push --tags
	git push
	rm -f dist/*
	rm -rf src/*.egg-info
	python3 setup.py sdist
	twine upload --skip-existing --verbose dist/*
