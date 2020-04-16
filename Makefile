
dev_install:
	pip install -e .[test]

test: type_check
	pytest -W ignore

type_check:
	mypy --ignore-missing-imports tfserver
