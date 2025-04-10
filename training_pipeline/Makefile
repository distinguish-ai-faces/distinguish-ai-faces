.PHONY: clean lint format test

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
	flake8 . --exclude=notebooks/
	pylint --recursive=y . --ignore=notebooks/
	mypy . --exclude notebooks/

lint-notebooks:
	nbqa flake8 notebooks/ --config=notebooks/.flake8
	nbqa black notebooks/ --check
	nbqa isort notebooks/ --check-only

format:
	black .
	isort .

format-notebooks:
	nbqa black notebooks/
	nbqa isort notebooks/

test:
	pytest -v tests/unit/
	pytest -v tests/integration/

test-coverage:
	pytest --cov=. --cov-report=term-missing --cov-report=xml

precommit:
	pre-commit run --all-files

install-dev:
	pip install -r requirements.txt
	pip install -e .
	pre-commit install 