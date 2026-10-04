.PHONY: install test lint cohort cohort-demo
	
install:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check . --fix

cohort:
	python -c "from dementia_undercoding.cohort import build, summarise; summarise(build())"

cohort-demo:
	python -c "from dementia_undercoding.cohort import build, summarise; summarise(build('config/config.demo.yaml'))"