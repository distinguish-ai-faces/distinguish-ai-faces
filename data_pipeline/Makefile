.PHONY: clean lint format test test-unit test-integration test-coverage

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".coverage" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name "dist" -exec rm -rf {} +
	find . -type d -name "build" -exec rm -rf {} +

lint:
	flake8 .
	pylint --recursive=y .
	mypy .

format:
	black .
	isort .

test:
	PYTHONPATH=. pytest tests/

test-unit:
	PYTHONPATH=. pytest tests/ -k "not integration"

test-integration:
	PYTHONPATH=. pytest tests/ -k "integration"

test-coverage:
	PYTHONPATH=. pytest tests/ --cov=src --cov-report=term-missing --cov-report=xml

precommit:
	pre-commit run --all-files

install-dev:
	pip install -r requirements.txt
	pip install -e .
	pre-commit install 