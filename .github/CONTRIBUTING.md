# Contributing to FastAPI REST Template

Thank you for your interest in contributing to the FastAPI REST Template! This document provides guidelines and instructions for contributing to this project.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Contributing Guidelines](#contributing-guidelines)
- [Development Workflow](#development-workflow)
- [Code Standards](#code-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [Release Process](#release-process)

## 🤝 Code of Conduct

This project adheres to a code of conduct that ensures a welcoming environment for all contributors. Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md) before contributing.

## 🚀 Getting Started

### Prerequisites

- Python 3.11 or higher
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- Git
- Basic understanding of FastAPI, Pydantic, and async Python

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/template-fastapi-rest.git
   cd template-fastapi-rest
   ```
3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/ORIGINAL_OWNER/template-fastapi-rest.git
   ```

##  Development Setup

### 1. Environment Setup

```bash
# Install dependencies using uv (recommended)
uv sync --dev

# Or using pip
pip install -e ".[dev]"
```

### 2. Environment Configuration

Create a `.env` file for development:

```bash
# Development Configuration
ENVIRONMENT=development
APP_DEBUG=true
APP_NAME="FastAPI REST Template (Dev)"

# Server Configuration
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
SERVER_RELOAD=true

# Logging
LOG_LEVEL=DEBUG

# API Keys (optional for development)
OPENAI_API_KEY=your_test_key
ANTHROPIC_API_KEY=your_test_key
GOOGLE_API_KEY=your_test_key
CUSTOM_API_KEY=your_test_key

# JWT (for testing)
JWT_SECRET_KEY=dev_secret_key_change_in_production
```

### 3. Pre-commit Hooks

Install pre-commit hooks for code quality:

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

### 4. Verify Setup

```bash
# Run tests to verify setup
pytest

# Start the development server
python app.py

# Visit http://localhost:8000/docs to verify the API
```

## 📁 Project Structure

Understanding the project structure is crucial for effective contributions:

```
template-fastapi-rest/
├── app.py                     # FastAPI application entry point
├── pyproject.toml            # Project configuration and dependencies
├── src/
│   ├── api/                  # API layer (routes, handlers, models, schemas)
│   ├── core/                 # Core application logic (managers, process management)
│   ├── services/             # Business logic services and pipelines
│   └── utils/                # Utilities (config, resources, io, auth)
├── runtime/                  # Application data directory
├── tests/                    # Test suite (unit, integration, e2e)
└── docs/                     # Documentation
```
<code_block_to_apply_changes_from>
```
tests/
├── conftest.py           # Test configuration and fixtures
├── unit/                 # Unit tests
│   ├── test_config.py    # Configuration tests
│   ├── test_process_manager.py
│   └── test_server_manager.py
├── integration/          # Integration tests
│   ├── test_api.py       # API integration tests
│   └── test_services.py  # Service integration tests
└── e2e/                  # End-to-end tests
    └── test_workflows.py # Complete workflow tests
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test types
pytest tests/unit/
pytest tests/integration/
pytest tests/e2e/

# Run with coverage
pytest --cov=src --cov-report=html tests/

# Run specific test file
pytest tests/unit/test_process_manager.py

# Run with verbose output
pytest -v
```

### Test Requirements

- **Unit Tests**: Test individual functions and classes in isolation
- **Integration Tests**: Test component interactions
- **E2E Tests**: Test complete workflows
- **Coverage**: Maintain at least 80% code coverage
- **Performance Tests**: Include performance benchmarks for critical paths

### Test Naming

- **Test Files**: `test_*.py`
- **Test Functions**: `test_*` or `*_test`
- **Test Classes**: `Test*`

Example:

```python
class TestProcessManager:
    """Test cases for ProcessManager class."""
    
    async def test_create_process_success(self):
        """Test successful process creation."""
        # Test implementation
        
    async def test_create_process_invalid_type(self):
        """Test process creation with invalid type."""
        # Test implementation
```

##  Documentation

### Documentation Types

1. **API Documentation**: Auto-generated from code docstrings
2. **User Guides**: How-to guides and tutorials
3. **Developer Documentation**: Architecture and development guides
4. **Configuration Documentation**: Configuration options and examples

### Documentation Standards

- Use clear, concise language
- Include code examples
- Keep documentation up-to-date with code changes
- Use proper markdown formatting

### Updating Documentation

When making changes that affect:

- **API endpoints**: Update route documentation and examples
- **Configuration**: Update configuration documentation
- **Architecture**: Update architecture diagrams and descriptions
- **Dependencies**: Update installation and setup instructions

## 🔀 Pull Request Process

### Before Submitting

1. **Run Tests**: Ensure all tests pass
   ```bash
   pytest
   ```

2. **Code Quality**: Run linting and formatting
   ```bash
   black src/ tests/
   isort src/ tests/
   flake8 src/ tests/
   mypy src/
   ```

3. **Update Documentation**: Update relevant documentation

4. **Test Your Changes**: Test your changes thoroughly

### Pull Request Template

When creating a pull request, please include:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing completed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests pass
- [ ] No breaking changes (or documented)

## Related Issues
Closes #123
```

### Review Process

1. **Automated Checks**: CI/CD pipeline runs tests and quality checks
2. **Code Review**: At least one maintainer reviews the code
3. **Testing**: Changes are tested in a staging environment
4. **Approval**: Maintainer approves and merges the PR

## 🐛 Issue Reporting

### Before Creating an Issue

1. Search existing issues to avoid duplicates
2. Check if the issue is already fixed in the latest version
3. Gather relevant information

### Issue Template

```markdown
## Bug Report / Feature Request

### Description
Clear description of the issue or feature request

### Environment
- OS: [e.g., macOS 13.0, Ubuntu 22.04]
- Python Version: [e.g., 3.11.0]
- FastAPI Version: [e.g., 0.104.1]

### Steps to Reproduce (for bugs)
1. Step 1
2. Step 2
3. Step 3

### Expected Behavior
What you expected to happen

### Actual Behavior
What actually happened

### Additional Context
Any additional information, logs, or screenshots
```

### Issue Labels

- **bug**: Something isn't working
- **enhancement**: New feature or request
- **documentation**: Improvements or additions to documentation
- **good first issue**: Good for newcomers
- **help wanted**: Extra attention is needed
- **priority: high**: High priority issue
- **priority: low**: Low priority issue

## 🚀 Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality in a backwards compatible manner
- **PATCH**: Backwards compatible bug fixes

### Release Checklist

1. **Update Version**: Update version in `pyproject.toml`
2. **Update Changelog**: Update `CHANGELOG.md`
3. **Run Tests**: Ensure all tests pass
4. **Update Documentation**: Update relevant documentation
5. **Create Release**: Create GitHub release with release notes

### Release Notes

Include in release notes:

- New features
- Bug fixes
- Breaking changes
- Migration guide (if needed)
- Contributors

## 🏷️ Commit Message Convention

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### Types

- **feat**: A new feature
- **fix**: A bug fix
- **docs**: Documentation only changes
- **style**: Changes that do not affect the meaning of the code
- **refactor**: A code change that neither fixes a bug nor adds a feature
- **perf**: A code change that improves performance
- **test**: Adding missing tests or correcting existing tests
- **chore**: Changes to the build process or auxiliary tools

### Examples

```
feat(process): add support for custom process types

fix(api): resolve timeout issue in process creation

docs: update configuration examples

refactor(config): simplify configuration loading logic
```

## 🤔 Getting Help

### Resources

- **Documentation**: Check the README.md and inline documentation
- **Issues**: Search existing issues or create a new one
- **Discussions**: Use GitHub Discussions for questions
- **Code Review**: Ask questions in pull request comments

### Contact

- **Maintainers**: @maintainer-username
- **Community**: GitHub Discussions
- **Security**: security@example.com (for security issues)

## 🎉 Recognition

Contributors will be recognized in:

- **README.md**: Contributors section
- **Release Notes**: Individual contributions
- **GitHub**: Contributor statistics

Thank you for contributing to the FastAPI REST Template! 🚀