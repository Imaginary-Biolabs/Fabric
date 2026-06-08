.PHONY: install lint test build publish-test publish clean

install:
	python -m pip install --upgrade pip
	if [ -d ../grumpy ]; then pip install -e ../grumpy; fi
	pip install -e ".[dev]"

lint:
	ruff check src tests

test:
	pytest tests/tolls/test_phase_1.py tests/tolls/test_phase_2.py tests/tolls/test_phase_3.py tests/tolls/test_metrics.py tests/tolls/test_phase_4.py tests/tolls/test_phase_5.py -x --tb=short -q

build:
	python -m pip install --upgrade build
	python -m build

publish-test: build
	python -m pip install --upgrade twine
	twine upload --repository testpypi dist/*

publish: build
	twine upload dist/*

clean:
	rm -rf build dist .pytest_cache .ruff_cache src/*.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
