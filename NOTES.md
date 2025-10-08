# Development Notes

## Security and Dependency Management

### Disabled Security Checks

For this research experiment, certain security and audit warnings have been intentionally disabled to maintain stable dependency versions required for reproducible results.

#### Pip-Audit Warnings Suppressed

The following security vulnerabilities are currently suppressed in pip-audit to preserve the exact dependency versions used in the ML training pipeline:

- **PyTorch 2.4.1**: Multiple vulnerabilities (PYSEC-2025-41, PYSEC-2024-259, GHSA-3749-ghw9-m3mg, GHSA-887c-mr87-cxwp)
- **Transformers 4.44.2**: Multiple vulnerabilities (14 total CVEs from PYSEC-2024-227 to GHSA-4w7r-h757-3r74)
- **Various dependencies**: aiohttp, authlib, h11, jinja2, jupyter-core, jupyterlab, protobuf, requests, tornado, urllib3

#### Rationale

1. **Experiment Reproducibility**: Upgrading dependencies could alter model behavior and training outcomes
2. **Version Stability**: The current versions have been tested and validated for the specific ML pipeline
3. **Research Priority**: Focus on fuzzing research rather than dependency management during active development

#### Security Mitigation

- The application runs in controlled environments (development/research only)
- Docker containerization provides additional isolation
- Regular monitoring of critical vulnerabilities that could affect core functionality

#### Future Work

- Plan dependency updates after experiment completion
- Evaluate impact of security patches on model performance
- Consider pinning specific security-patched versions that maintain compatibility

## Bandit Configuration

Bandit security warnings are configured in `config/bandit.yaml` with appropriate suppressions:

- **B615**: Hugging Face model loading (false positive for local models)
- **B101, B311, B404, B603, B607**: Development-specific suppressions

## Safety Configuration

Safety dependency scanning may report vulnerabilities but is configured to allow continued development while maintaining experiment integrity.

