# Contributing to PDE-Fluid-Φ

We welcome contributions to PDE-Fluid-Φ! This document provides guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)
- [Issue Reporting](#issue-reporting)
- [Community](#community)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

1. **Fork the repository** and clone your fork
2. **Set up the development environment** (see Development Setup)
3. **Find an issue** to work on or propose a new feature
4. **Create a branch** for your changes
5. **Make your changes** following our guidelines
6. **Submit a pull request**

## Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- Docker (optional, for containerized development)
- CUDA toolkit (optional, for GPU development)

### Installation

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/pde-fluid-phi.git
cd pde-fluid-phi

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Docker Development Environment

```bash
# Build development container
docker build --target development -t pde-fluid-phi:dev .

# Run development container
docker run -it --rm \
    -v $(pwd):/app \
    -p 8888:8888 \
    pde-fluid-phi:dev
```

### Verify Installation

```bash
# Run tests
pytest tests/

# Check code quality
black --check src/ tests/
isort --check-only src/ tests/
flake8 src/ tests/
mypy src/

# Run example
python examples/basic_usage.py
```

## Contributing Guidelines

### Types of Contributions

We welcome several types of contributions:

- **Bug fixes**: Fix issues in the codebase
- **Feature development**: Add new functionality
- **Documentation**: Improve docs, examples, tutorials
- **Performance optimization**: Improve speed or memory usage
- **Testing**: Add or improve test coverage
- **Infrastructure**: CI/CD, deployment, tooling

### Contribution Workflow

1. **Check existing issues** to avoid duplicate work
2. **Open an issue** to discuss significant changes
3. **Fork and clone** the repository
4. **Create a feature branch**: `git checkout -b feature/your-feature-name`
5. **Make your changes** with proper commits
6. **Add tests** for new functionality
7. **Update documentation** as needed
8. **Run quality checks** locally
9. **Push changes** and create a pull request

### Commit Message Guidelines

We follow the [Conventional Commits](https://conventionalcommits.org/) specification:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `chore`: Maintenance tasks
- `ci`: CI/CD changes

**Examples:**
```
feat(operators): add rational Fourier operator with stability constraints

fix(training): resolve memory leak in distributed training

docs(readme): update installation instructions for CUDA support

test(operators): add comprehensive tests for spectral layers
```

## Pull Request Process

### Before Submitting

1. **Ensure all tests pass**: `pytest tests/`
2. **Check code quality**: Run linting and formatting
3. **Update documentation**: Add docstrings and update docs
4. **Add tests**: Ensure good test coverage
5. **Update CHANGELOG.md**: Document your changes

### Pull Request Template

When creating a pull request, please use our template:

```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that causes existing functionality to change)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
- [ ] I have added tests that prove my fix is effective or feature works
- [ ] New and existing unit tests pass locally
- [ ] I have tested the changes on relevant hardware (CPU/GPU)

## Checklist
- [ ] My code follows the project's style guidelines
- [ ] I have performed a self-review of my code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] Any dependent changes have been merged and published

## Screenshots/Demos (if applicable)
Add screenshots or demo videos for UI changes or new features.

## Additional Notes
Any additional information, concerns, or questions.
```

### Review Process

1. **Automated checks** must pass (CI/CD pipeline)
2. **Code review** by maintainers
3. **Discussion** and potential changes
4. **Approval** and merge

## Code Style

### Python Code Style

We use the following tools for code quality:

- **Black**: Code formatting
- **isort**: Import sorting  
- **flake8**: Linting
- **mypy**: Type checking

### Configuration

See `pyproject.toml` for detailed configuration.

### Key Guidelines

- **Line length**: 100 characters maximum
- **Type hints**: Required for all public functions
- **Docstrings**: Google-style docstrings for all public functions
- **Imports**: Sorted and grouped properly
- **Naming**: Snake_case for variables and functions, PascalCase for classes

### Example Code Style

```python
"""Module docstring describing the module's purpose."""

import torch
import numpy as np
from typing import Tuple, Optional

from ..utils.spectral_utils import compute_energy_spectrum


class RationalFourierOperator(nn.Module):
    """
    Rational Fourier Neural Operator for turbulent flow modeling.
    
    This operator uses rational function approximations in Fourier space
    to achieve numerical stability for high Reynolds number flows.
    
    Args:
        modes: Number of Fourier modes in each dimension
        width: Hidden dimension width
        n_layers: Number of operator layers
        
    Example:
        >>> operator = RationalFourierOperator(modes=(32, 32, 32), width=64)
        >>> output = operator(input_tensor)
    """
    
    def __init__(
        self,
        modes: Tuple[int, int, int],
        width: int = 64,
        n_layers: int = 4
    ) -> None:
        super().__init__()
        self.modes = modes
        self.width = width
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the operator.
        
        Args:
            x: Input tensor of shape [batch, channels, height, width, depth]
            
        Returns:
            Output tensor with same shape as input
            
        Raises:
            ValueError: If input dimensions are incorrect
        """
        if x.dim() != 5:
            raise ValueError(f"Expected 5D input, got {x.dim()}D")
        
        # Implementation here
        return x
```

## Testing

### Test Structure

- **Unit tests**: Test individual functions and classes
- **Integration tests**: Test component interactions
- **End-to-end tests**: Test complete workflows
- **Performance tests**: Benchmark critical paths
- **GPU tests**: Test CUDA kernels and GPU operations

### Writing Tests

```python
import pytest
import torch
from pde_fluid_phi.operators import RationalFourierOperator


class TestRationalFourierOperator:
    """Test suite for RationalFourierOperator."""
    
    def test_initialization(self):
        """Test operator initialization."""
        operator = RationalFourierOperator(modes=(16, 16, 16))
        assert operator.modes == (16, 16, 16)
        assert operator.width == 64  # default value
    
    def test_forward_pass(self):
        """Test forward pass with valid input."""
        operator = RationalFourierOperator(modes=(16, 16, 16))
        x = torch.randn(2, 3, 32, 32, 32)
        
        output = operator(x)
        
        assert output.shape == x.shape
        assert torch.isfinite(output).all()
    
    def test_invalid_input_shape(self):
        """Test error handling for invalid input shapes."""
        operator = RationalFourierOperator(modes=(16, 16, 16))
        x = torch.randn(2, 3, 32, 32)  # 4D instead of 5D
        
        with pytest.raises(ValueError, match="Expected 5D input"):
            operator(x)
    
    @pytest.mark.gpu
    def test_gpu_compatibility(self):
        """Test GPU compatibility if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        operator = RationalFourierOperator(modes=(16, 16, 16)).cuda()
        x = torch.randn(2, 3, 32, 32, 32, device='cuda')
        
        output = operator(x)
        assert output.device.type == 'cuda'
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_operators.py

# Run tests with coverage
pytest --cov=src --cov-report=html

# Run only GPU tests
pytest -m gpu

# Run tests in parallel
pytest -n auto
```

## Documentation

### Types of Documentation

- **API Documentation**: Docstrings in code
- **User Guide**: High-level usage documentation
- **Tutorials**: Step-by-step guides
- **Examples**: Practical use cases
- **Architecture Documentation**: Technical design docs

### Writing Documentation

#### Docstrings

Use Google-style docstrings:

```python
def compute_energy_spectrum(
    velocity: torch.Tensor,
    return_wavenumbers: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Compute radially averaged energy spectrum from velocity field.
    
    The energy spectrum E(k) represents the distribution of kinetic energy
    across different wavenumbers, providing insight into turbulence structure.
    
    Args:
        velocity: Velocity field tensor of shape [batch, 3, h, w, d] where
            the second dimension contains [u, v, w] velocity components.
        return_wavenumbers: If True, return both spectrum and wavenumber
            array. If False, return only spectrum.
            
    Returns:
        If return_wavenumbers is False:
            Energy spectrum array of shape [batch, n_wavenumbers]
        If return_wavenumbers is True:
            Tuple of (energy_spectrum, wavenumbers) where wavenumbers
            has shape [n_wavenumbers]
            
    Raises:
        ValueError: If velocity tensor doesn't have exactly 3 velocity
            components in the second dimension.
            
    Example:
        >>> velocity = torch.randn(2, 3, 64, 64, 64)
        >>> spectrum = compute_energy_spectrum(velocity)
        >>> spectrum.shape
        torch.Size([2, 32])
        
        >>> spectrum, k = compute_energy_spectrum(velocity, return_wavenumbers=True)
        >>> k.shape
        torch.Size([32])
    
    Note:
        The radial averaging assumes isotropic turbulence. For anisotropic
        flows, consider using directional spectra instead.
    """
```

#### Markdown Documentation

- Use clear headings and structure
- Include code examples
- Add diagrams where helpful
- Keep language accessible

### Building Documentation

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build HTML documentation
cd docs/
make html
```

## Issue Reporting

### Before Reporting

1. **Search existing issues** to avoid duplicates
2. **Check documentation** for solutions
3. **Try the latest version**
4. **Prepare a minimal reproducible example**

### Issue Template

```markdown
## Bug Report / Feature Request

**Type**: Bug Report / Feature Request / Question

## Description
Clear description of the issue or feature request.

## Environment
- OS: [e.g., Ubuntu 20.04, macOS 12.0, Windows 11]
- Python version: [e.g., 3.9.5]
- PDE-Fluid-Φ version: [e.g., 0.1.0]
- PyTorch version: [e.g., 2.0.1]
- CUDA version: [e.g., 11.8] (if applicable)
- GPU: [e.g., NVIDIA RTX 4090] (if applicable)

## Steps to Reproduce
1. Step 1
2. Step 2
3. Step 3

## Expected Behavior
What you expected to happen.

## Actual Behavior
What actually happened.

## Code Example
```python
# Minimal reproducible example
import pde_fluid_phi
# ... code that reproduces the issue
```

## Error Messages/Logs
```
Paste any error messages or relevant logs here
```

## Additional Context
Any other context about the problem.
```

## Community

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and discussions
- **Email**: research@terragonlabs.com for sensitive issues

### Getting Help

1. **Check documentation** first
2. **Search existing issues** and discussions
3. **Ask questions** in GitHub Discussions
4. **Join our community** for real-time help

### Contributing to Community

- **Answer questions** from other users
- **Review pull requests** 
- **Share examples** and tutorials
- **Report bugs** you encounter
- **Suggest improvements**

## Recognition

We value all contributions and recognize contributors in several ways:

- **Contributors file**: Listed in CONTRIBUTORS.md
- **Release notes**: Significant contributions mentioned
- **GitHub**: Contributor badge and stats
- **Documentation**: Attribution in relevant sections

## License

By contributing to PDE-Fluid-Φ, you agree that your contributions will be licensed under the MIT License.

## Questions?

If you have questions about contributing, please:

1. Check this document and other documentation
2. Search existing GitHub issues and discussions
3. Open a new discussion in GitHub Discussions
4. Contact us at research@terragonlabs.com

Thank you for contributing to PDE-Fluid-Φ! 🌊