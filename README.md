# NeoFuzz

A framework for generating and evaluating fuzzing inputs for the Lua interpreter using large language models.

## About

NeoFuzz leverages GPT Neo to learn the characteristics of Lua code from large-scale datasets using Self Supervised Learning (SSL), and applies Proximal Policy Optimization (PPO) to further fine-tune the model for generating syntactically and semantically valid Lua scripts.

The goal is to maximize code coverage and discover edge cases in the Lua interpreter by producing diverse and effective fuzzing inputs through reinforcement learning and language modeling.

## Quick Start

Follow the quick setup guide in the [Jupyter Notebook](neo_fuzz.ipynb)


## Workflow

1. **Data Preprocessing**: Initialize and preprocess training data
2. **SSL Training**: Initial self-supervised learning for Lua code generation
3. **Evaluation**: Assess baseline validity of generated samples
4. **PPO Training**: Fine-tune with reinforcement learning for better semantic correctness
5. **Evaluation**: Assess validity against SSL baseline
6. **AFL Evaluation**: Fuzz testing with AFL++ and coverage analysis
7. **GPT Neo Evaluation**: Fuzz testing with GPT Neo and coverage analysis

## Development

### Available Commands

Run `make help` to see all available commands:

- **Development**: Setup and development tools
- **Code Quality**: Linting and formatting (black, flake8, isort, mypy)
- **Security**: Security checks (bandit, safety, pip-audit)
- **Maintenance**: Cleanup and utilities

### Code Quality

The project uses several tools to maintain code quality:

- **Black**: Code formatting
- **Flake8**: Style guide enforcement
- **isort**: Import sorting
- **MyPy**: Type checking
- **Bandit**: Security linting
- **Pre-commit**: Automated checks before commits

### Configuration

Configuration files are located in the `config/` directory:

- `pyproject.toml`: Black, isort, and MyPy configuration
- `setup.cfg`: Flake8 configuration
- `bandit.yaml`: Security linting configuration
- `path.env`: Path configuration for application components
- `requirements/`: Dependency specifications

## License

See [LICENSE](LICENSE) for details.
