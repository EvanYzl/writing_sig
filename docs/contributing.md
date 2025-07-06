# Contributing to MSA-T OSV

Thank you for your interest in contributing to MSA-T OSV! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Release Process](#release-process)

## Code of Conduct

This project adheres to the Contributor Covenant Code of Conduct. By participating, you are expected to uphold this code.

## Getting Started

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/msa-t-osv.git
   cd msa-t-osv
   ```

3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/original-owner/msa-t-osv.git
   ```

### Development Setup

1. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install development dependencies**:
   ```bash
   pip install -r requirements-dev.txt
   pip install -e .
   ```

3. **Setup pre-commit hooks**:
   ```bash
   pre-commit install
   ```

## Contributing Guidelines

### Types of Contributions

We welcome various types of contributions:

- **Bug fixes**: Fix bugs and issues
- **Feature additions**: Add new functionality
- **Documentation**: Improve or add documentation
- **Tests**: Add or improve test coverage
- **Performance improvements**: Optimize code performance
- **Code quality**: Improve code structure and readability

### Before Contributing

1. **Check existing issues**: Search for existing issues or pull requests
2. **Create an issue**: For new features or major changes, create an issue first
3. **Discuss**: Engage in discussions on issues before implementing

### Branch Strategy

- **main**: Production-ready code
- **develop**: Integration branch for features
- **feature/***: Feature branches
- **bugfix/***: Bug fix branches
- **hotfix/***: Critical bug fixes

### Creating a Feature Branch

```bash
# Update your local main branch
git checkout main
git pull upstream main

# Create and switch to a new feature branch
git checkout -b feature/your-feature-name

# Make your changes and commit
git add .
git commit -m "Add feature: description of changes"
```

## Code Style

### Python Style Guide

We follow PEP 8 with some modifications:

- **Line length**: 127 characters maximum
- **Import sorting**: Use isort
- **Code formatting**: Use Black
- **Type hints**: Use mypy-compatible type hints

### Code Formatting

```bash
# Format code with Black
black msa_t_osv/

# Sort imports with isort
isort msa_t_osv/

# Check code style with flake8
flake8 msa_t_osv/

# Type checking with mypy
mypy msa_t_osv/
```

### Naming Conventions

- **Classes**: PascalCase (e.g., `MSATOSVModel`)
- **Functions and variables**: snake_case (e.g., `compute_eer`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `DEFAULT_CONFIG`)
- **Files**: snake_case (e.g., `model_utils.py`)

### Documentation Standards

- **Docstrings**: Use Google style docstrings
- **Comments**: Write clear, concise comments
- **README**: Keep updated with new features

Example docstring:
```python
def compute_eer(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute Equal Error Rate (EER).
    
    Args:
        labels: Ground truth labels (0 for genuine, 1 for forged)
        scores: Predicted scores (higher values indicate forged)
        
    Returns:
        Equal Error Rate as a float between 0 and 1
        
    Raises:
        ValueError: If inputs have different lengths or invalid values
    """
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=msa_t_osv --cov-report=html

# Run specific test file
pytest tests/test_models.py

# Run with verbose output
pytest -v
```

### Writing Tests

- **Test coverage**: Aim for >90% coverage
- **Test organization**: Group related tests in classes
- **Test naming**: Use descriptive test names
- **Fixtures**: Use pytest fixtures for common setup

Example test:
```python
def test_model_forward_pass(sample_config, sample_batch):
    """Test model forward pass with valid inputs."""
    model = MSATOSVModel(sample_config)
    logits = model(sample_batch)
    
    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (sample_batch.shape[0], 2)
    assert not torch.isnan(logits).any()
```

### Test Guidelines

1. **Unit tests**: Test individual functions and classes
2. **Integration tests**: Test component interactions
3. **Edge cases**: Test boundary conditions and error cases
4. **Performance tests**: Test for performance regressions

## Documentation

### Documentation Structure

```
docs/
├── api.md              # API documentation
├── installation.md     # Installation guide
├── contributing.md     # This file
├── configuration.md    # Configuration guide
├── training.md         # Training guide
├── evaluation.md       # Evaluation guide
└── examples/           # Code examples
```

### Writing Documentation

- **Clear and concise**: Write for different skill levels
- **Examples**: Include code examples
- **Cross-references**: Link related sections
- **Keep updated**: Update docs with code changes

### Building Documentation

```bash
# Install documentation dependencies
pip install sphinx sphinx-rtd-theme myst-parser

# Build documentation
cd docs
make html

# View documentation
open _build/html/index.html
```

## Pull Request Process

### Before Submitting

1. **Test your changes**:
   ```bash
   pytest tests/
   flake8 msa_t_osv/
   mypy msa_t_osv/
   ```

2. **Update documentation**: Add or update relevant documentation

3. **Add tests**: Include tests for new functionality

4. **Check formatting**: Ensure code follows style guidelines

### Pull Request Template

Use this template for pull requests:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
- [ ] Tests added/updated
- [ ] All tests pass
- [ ] Manual testing completed

## Documentation
- [ ] Documentation updated
- [ ] Examples added/updated

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Corresponding issue created/updated
```

### Review Process

1. **Automated checks**: CI/CD pipeline runs tests and checks
2. **Code review**: At least one maintainer reviews the PR
3. **Address feedback**: Respond to review comments
4. **Merge**: PR is merged after approval

## Release Process

### Versioning

We use semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Breaking changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

### Release Steps

1. **Update version**: Update version in `setup.py` and `__init__.py`
2. **Update changelog**: Add release notes to `CHANGELOG.md`
3. **Create release branch**: `git checkout -b release/v1.0.0`
4. **Final testing**: Run full test suite
5. **Create tag**: `git tag -a v1.0.0 -m "Release v1.0.0"`
6. **Merge to main**: Merge release branch to main
7. **Publish**: Push tag and create GitHub release

## Getting Help

### Communication Channels

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and discussions
- **Email**: For private or sensitive matters

### Asking Questions

When asking for help:

1. **Provide context**: Explain what you're trying to do
2. **Include code**: Share relevant code snippets
3. **Error messages**: Include full error messages
4. **Environment**: Mention your system and versions

## Recognition

Contributors will be recognized in:

- **README.md**: List of contributors
- **Release notes**: Credit for contributions
- **Documentation**: Attribution for major contributions

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.

## Questions?

If you have questions about contributing, please:

1. Check this document first
2. Search existing issues and discussions
3. Create a new issue or discussion
4. Contact the maintainers directly

Thank you for contributing to MSA-T OSV! 