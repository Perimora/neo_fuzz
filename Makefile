RED=\033[0;31m
GREEN=\033[0;32m
YELLOW=\033[1;33m
BLUE=\033[0;34m
NC=\033[0m 

ifneq (,$(wildcard config/.env))
	include config/.env
	export
endif

VENV_ROOT=.venv
VENV_BIN=$(VENV_ROOT)/bin

.PHONY: help install-dev install-pre-commit pre-commit lint format security bandit safety pip-audit test-unit test-integration test-e2e test-fast test test-coverage test-coverage-unit test-coverage-integration test-coverage-e2e coverage-report coverage-html coverage-xml coverage-check coverage-clean test-all test-ci clean help

install-dev:
	sudo apt update && sudo apt install -y python3 python3-venv python3-pip
	@if [ ! -d "$(VENV_ROOT)" ]; then \
		python3 -m venv $(VENV_ROOT) && \
		echo "Virtual environment created at $(VENV_ROOT)"; \
	else \
		echo "Using existing virtual environment at $(VENV_ROOT)"; \
	fi
	$(VENV_ROOT)/bin/pip install --upgrade pip
	$(VENV_ROOT)/bin/pip install -r requirements-dev.txt
	@echo "$(GREEN)[+] Development environment is ready. Activate it with: source $(VENV_BIN)/activate"

install-pre-commit:
	$(VENV_BIN)/pre-commit install

pre-commit:
	$(VENV_BIN)/pre-commit run --all-files

lint:
	$(VENV_BIN)/black --check --diff --config config/pyproject.toml .
	$(VENV_BIN)/flake8 --config config/setup.cfg
	$(VENV_BIN)/isort --check --diff --settings-path config/pyproject.toml src test
	$(VENV_BIN)/mypy --config-file config/setup.cfg src test

format:
	$(VENV_BIN)/black --config config/pyproject.toml .
	$(VENV_BIN)/isort --settings-path config/pyproject.toml src test

security: bandit safety pip-audit

bandit:
	$(VENV_BIN)/bandit -r src -x test -c config/bandit.yaml

safety:
	SAFETY_API_KEY=$(SAFETY_API_KEY) $(VENV_BIN)/safety scan --full-report --file=requirements-dev.txt

pip-audit:
	$(VENV_BIN)/pip-audit

test-unit:
	@echo "$(BLUE)Running unit tests...$(NC)"
	$(VENV_BIN)/pytest test/unit -v --tb=short

test-integration:
	@echo "$(BLUE)Running integration tests...$(NC)"
	$(VENV_BIN)/pytest test/integration -v --tb=short

test-e2e:
	@echo "$(BLUE)Running end-to-end tests...$(NC)"
	$(VENV_BIN)/pytest test/e2e -v --tb=short

test-fast: test-unit
	@echo "$(GREEN)Fast tests completed!$(NC)"

test: test-unit test-integration test-e2e
	@echo "$(GREEN)All tests completed!$(NC)"

test-coverage:
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	$(VENV_BIN)/coverage run --rcfile=config/.coveragerc -m pytest

test-coverage-unit:
	@echo "$(BLUE)Running unit tests with coverage...$(NC)"
	$(VENV_BIN)/coverage run --rcfile=config/.coveragerc -m pytest test/unit

test-coverage-integration:
	@echo "$(BLUE)Running integration tests with coverage...$(NC)"
	$(VENV_BIN)/coverage run --rcfile=config/.coveragerc -m pytest test/integration

test-coverage-e2e:
	@echo "$(BLUE)Running e2e tests with coverage...$(NC)"
	$(VENV_BIN)/coverage run --rcfile=config/.coveragerc -m pytest test/e2e

coverage-report: coverage-html coverage-xml

coverage-html:
	$(VENV_BIN)/coverage html --rcfile=config/.coveragerc

coverage-xml:
	$(VENV_BIN)/coverage xml --rcfile=config/.coveragerc

coverage-check:
	$(VENV_BIN)/coverage report --fail-under=80 --rcfile=config/.coveragerc

coverage-clean:
	rm -rf reports/coverage

test-all: lint security test-coverage coverage-check
	@echo "$(GREEN)All checks and tests passed!$(NC)"

test-ci: test-unit test-integration
	@echo "$(GREEN)CI tests completed!$(NC)"

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .coverage .coverage.* .tox .eggs *.egg-info build dist .venv reports/ logs/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true


help:
	@echo "$(BLUE)Available commands:$(NC)"
	@echo "$(GREEN)Development:$(NC)"
	@echo "  install-dev       - Install development dependencies"
	@echo "  install-pre-commit - Install pre-commit hooks"
	@echo "  pre-commit        - Run pre-commit hooks"
	@echo ""
	@echo "$(GREEN)Code Quality:$(NC)"
	@echo "  lint              - Run all linting tools (black, flake8, isort, mypy)"
	@echo "  format            - Format code with black and isort"
	@echo "  security          - Run security scans (bandit, safety, pip-audit)"
	@echo ""
	@echo "$(GREEN)Testing:$(NC)"
	@echo "  test-unit         - Run unit tests only"
	@echo "  test-integration  - Run integration tests only"
	@echo "  test-e2e          - Run end-to-end tests only"
	@echo "  test-fast         - Run unit tests (alias for test-unit)"
	@echo "  test              - Run all tests (unit, integration, e2e)"
	@echo "  test-coverage     - Run all tests with coverage"
	@echo "  test-all          - Run lint, security, and coverage tests"
	@echo "  test-ci           - Run unit and integration tests (for CI)"
	@echo ""
	@echo "$(GREEN)Coverage:$(NC)"
	@echo "  coverage-report   - Generate coverage reports (HTML and XML)"
	@echo "  coverage-html     - Generate HTML coverage report"
	@echo "  coverage-xml      - Generate XML coverage report"
	@echo "  coverage-check    - Check coverage threshold (80%)"
	@echo "  coverage-clean    - Clean coverage files"
	@echo ""
	@echo "$(GREEN)Maintenance:$(NC)"
	@echo "  clean             - Clean temporary files and caches"
	@echo "  help              - Show this help message"
