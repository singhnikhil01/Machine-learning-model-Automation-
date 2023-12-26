.PHONY: install test format lint all

PYTHON_FILES := $(shell find . -name "*.py")

format:
	black $(PYTHON_FILES)

install:
	pip install --upgrade pip && pip install -r requirements.txt

test:
	python -m pytest -vv --cov=app test_app.py

lint:
	pylint --disable=R,C $(PYTHON_FILES)

all: install test lint format
