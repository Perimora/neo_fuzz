# python_ci_cd

A modern Python project template with robust CI/CD, code quality, security tooling, and structured logging.

## Features
- **Development Environment**: Isolated `venv` with automated setup
- **Code Quality**: `black`, `flake8`, `isort`, `mypy` with comprehensive linting
- **Security**: `bandit`, `safety`, `pip-audit` for vulnerability scanning
- **Testing**: `pytest` with structured test organization (unit/integration/e2e)
- **Coverage**: Enforced coverage thresholds with HTML/XML reports
- **Logging**: Environment-aware structured logging with JSON configuration
- **Pre-commit**: Automated hooks for code quality enforcement
- **CI/CD**: GitHub Actions pipeline with caching and artifact uploads
- **Makefile**: Comprehensive automation for all development workflows

## Quickstart

### 0. Add your SafetyCLI API key
Since SafetyCLI now demands an existing API key please create a `.env` file in `config/`:

```bash
echo "SAFETY_API_KEY=$YOUR_API_KEY" > config/.env
```
> Note: If you don't add your API key the security checks and therefore also the pre-commit checks will fail. 

### 1. Clone and Install
```bash
make install-dev
source .venv/bin/activate
```

### 2. Code Quality
- Lint: `make lint`
- Format: `make format`
- Security: `make security`

### 3. Testing
- **Unit tests**: `make test-unit`
- **Integration tests**: `make test-integration`
- **End-to-end tests**: `make test-e2e`
- **Fast tests**: `make test-fast` (unit only)
- **All tests**: `make test`
- **Coverage**: `make test-coverage`
- **Coverage reports**: `make coverage-report`
- **Coverage check**: `make coverage-check` (80% threshold)

### 4. Logging
The template includes a structured logging system with:
- Environment-aware configuration (development/staging/production)
- Per-level log files (debug, info, warning, error, master)
- JSON-based configuration in `config/logging/`
- Automatic log directory creation

### 5. Pre-commit
- Install hooks: `make install-pre-commit`
- Run all hooks: `make pre-commit`

### 6. Cleaning
- Remove build/test artifacts: `make clean`

## CI/CD
- **Automated Pipeline**: Runs on push and PR via GitHub Actions
- **Dependency Caching**: Pip dependencies cached for faster builds
- **Quality Gates**: Enforces lint, security, and test coverage
- **Structured Jobs**: Separate lint, security, and test jobs with dependencies
- **Artifact Uploads**: Coverage reports uploaded as build artifacts
- **Coverage Enforcement**: 80% coverage threshold enforced in CI

## Project Structure
```
├── src/                          # Application code
│   ├── app.py                   # Main application
│   └── classes/                 # Application classes
│       └── log/                 # Logging system
│           ├── logger.py        # Main logger class
│           ├── logging_handler.py # Handler setup
│           ├── logging_level.py # Log level enum
│           └── logging_env.py   # Environment enum
├── test/                        # Test suite
│   ├── unit/                    # Unit tests
│   ├── integration/             # Integration tests
│   └── e2e/                     # End-to-end tests
├── config/                      # Configuration files
│   ├── pyproject.toml          # Black, isort, coverage config
│   ├── setup.cfg               # Flake8, mypy config
│   ├── bandit.yaml             # Security scan config
│   ├── .coveragerc             # Coverage config
│   └── logging/                # Logging configuration
│       └── logging_paths.json  # Environment-specific log paths
├── logs/                        # Generated log files
│   └── development/            # Environment-specific logs
├── reports/                     # Generated reports
│   └── coverage/               # Coverage reports
├── requirements-dev.txt         # Development dependencies
├── Makefile                    # Development automation
├── .pre-commit-config.yaml     # Pre-commit hooks
└── .github/workflows/ci.yml    # CI/CD pipeline
```

## Available Make Commands

Run `make help` to see all available commands:

### Development
- `make install-dev` - Install development dependencies
- `make install-pre-commit` - Install pre-commit hooks
- `make pre-commit` - Run pre-commit hooks

### Code Quality
- `make lint` - Run all linting tools
- `make format` - Format code with black and isort
- `make security` - Run security scans

### Testing
- `make test-unit` - Run unit tests only
- `make test-integration` - Run integration tests only
- `make test-e2e` - Run end-to-end tests only
- `make test-fast` - Run unit tests (alias)
- `make test` - Run all tests
- `make test-coverage` - Run all tests with coverage
- `make test-all` - Run lint, security, and coverage tests
- `make test-ci` - Run unit and integration tests (for CI)

### Coverage
- `make coverage-report` - Generate coverage reports
- `make coverage-html` - Generate HTML coverage report
- `make coverage-xml` - Generate XML coverage report
- `make coverage-check` - Check coverage threshold
- `make coverage-clean` - Clean coverage files

### Maintenance
- `make clean` - Clean temporary files and caches
- `make help` - Show help message
