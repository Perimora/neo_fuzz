# Security Considerations

## Known Vulnerabilities

### PyTorch CVE-2025-3730 (GHSA-887c-mr87-cxwp)
- **Component**: `torch.nn.functional.ctc_loss`
- **Version Affected**: PyTorch 2.6.0, 2.7.0, 2.7.1
- **Impact**: Denial of Service (local attack)
- **Status**: No fix available as of January 2025
- **Project Impact**: LOW - ctc_loss function not used in this codebase

#### Mitigation Strategy
1. **Current Approach**: Continue using PyTorch 2.7.1
2. **Rationale**: Vulnerability doesn't affect our use case (no CTC loss usage)
3. **Monitoring**: Check for updates monthly via `pip-audit`

#### Action Items
- [ ] Monitor PyTorch security advisories
- [ ] Update when patch becomes available
- [ ] Avoid using `torch.nn.functional.ctc_loss` until patched

## Security Scanning

### Regular Checks
```bash
# Run security checks (PyTorch vulnerability is already suppressed in Makefile)
make security

# Or run pip-audit individually
pip-audit --ignore-vuln GHSA-887c-mr87-cxwp

# Check for PyTorch updates
pip index versions torch
```

### Bandit Security Scanner
- Configuration: `config/bandit.yaml`
- Status: All warnings addressed with appropriate suppressions
- Last scan: [Update with current date]

## Dataset Security
- **HuggingFace Downloads**: Secured with revision pinning
- **Bandit Suppressions**: Applied for legitimate use cases
- **Revision Pinning**: Implemented for reproducible builds
