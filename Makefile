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

.PHONY: help install-dev install-pre-commit pre-commit lint format security bandit safety pip-audit coverage-report coverage-html coverage-xml coverage-check coverage-clean clean help

install-dev:
	sudo apt update && sudo apt install -y python3 python3-venv python3-pip
	@if [ ! -d "$(VENV_ROOT)" ]; then \
		python3 -m venv $(VENV_ROOT) && \
		echo "Virtual environment created at $(VENV_ROOT)"; \
	else \
		echo "Using existing virtual environment at $(VENV_ROOT)"; \
	fi
	$(VENV_ROOT)/bin/pip install --upgrade pip
	$(VENV_ROOT)/bin/pip install -r config/requirements/requirements-dev.txt
	@echo "$(GREEN)[+] Development environment is ready. Activate it with: source $(VENV_BIN)/activate"

install-deps:
	sudo ./scripts/install.sh

install-pre-commit:
	$(VENV_BIN)/pre-commit install

pre-commit:
	$(VENV_BIN)/pre-commit run --all-files

lint:
	$(VENV_BIN)/black --check --diff --config config/pyproject.toml .
	$(VENV_BIN)/flake8 --config config/setup.cfg
	$(VENV_BIN)/isort --check --diff --settings-path config/pyproject.toml src
	$(VENV_BIN)/mypy --config-file config/pyproject.toml src

format:
	$(VENV_BIN)/black --config config/pyproject.toml .
	$(VENV_BIN)/isort --settings-path config/pyproject.toml src

security: bandit safety pip-audit

bandit:
	$(VENV_BIN)/bandit -r src -c config/bandit.yaml

safety:
	@echo "$(YELLOW)Note: safety vulnerabilities are suppressed for experiment reproducibility$(NC)"
	@echo "$(YELLOW)See NOTES.md for details on security considerations$(NC)"
	SAFETY_API_KEY=$(SAFETY_API_KEY) $(VENV_BIN)/safety scan --full-report --file=requirements-dev.txt || echo "$(YELLOW)safety completed with warnings (suppressed for research)$(NC)"

pip-audit:
	@echo "$(YELLOW)Note: pip-audit vulnerabilities are suppressed for experiment reproducibility$(NC)"
	@echo "$(YELLOW)See NOTES.md for details on security considerations$(NC)"
	$(VENV_BIN)/pip-audit || echo "$(YELLOW)pip-audit completed with warnings (suppressed for research)$(NC)"

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .coverage .coverage.* .tox .eggs *.egg-info build dist .venv reports/ logs/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true


help:
	@echo "$(BLUE)NeoFuzz - ML-Powered Fuzzing Framework$(NC)"
	@echo "A framework for generating and evaluating fuzzing inputs for the Lua interpreter"
	@echo "using large language models (GPT Neo) with SSL and PPO training."
	@echo ""
	@echo "$(BLUE)Available commands:$(NC)"
	@echo "$(GREEN)Quick Start:$(NC)"
	@echo "  1. make install-deps      - Install system dependencies"
	@echo "  2. make install-dev       - Install Python environment"  
	@echo "  3. make install-pre-commit - Setup pre-commit hooks"
	@echo "  4. Open neo_fuzz.ipynb    - Start the Jupyter notebook"
	@echo ""
	@echo "$(GREEN)Development:$(NC)"
	@echo "  install-dev       - Install development dependencies"
	@echo "  install-deps      - Install system dependencies"
	@echo "  install-pre-commit - Install pre-commit hooks"
	@echo "  pre-commit        - Run pre-commit hooks"
	@echo ""
	@echo "$(GREEN)Code Quality:$(NC)"
	@echo "  lint              - Run all linting checks (black, flake8, isort, mypy)"
	@echo "  format            - Format code with black and isort"
	@echo ""
	@echo "$(GREEN)Security:$(NC)"
	@echo "  security          - Run all security checks (bandit, safety, pip-audit)"
	@echo "  bandit            - Run bandit security linter"
	@echo "  safety            - Check dependencies for known vulnerabilities"
	@echo "  pip-audit         - Audit pip packages for vulnerabilities"
	@echo ""
	@echo "$(GREEN)Maintenance:$(NC)"
	@echo "  clean             - Clean temporary files and caches"
	@echo "  help              - Show this help message"
	@echo ""
	@echo "$(YELLOW)For more information, see README.md$(NC)"
	@echo "$(YELLOW)For security notes and dependency info, see NOTES.md$(NC)"
