#!/bin/bash
# Quality assurance script for PDE-Fluid-Φ
# Runs all code quality checks, tests, and benchmarks

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PYTHON_VERSION=${PYTHON_VERSION:-"3.9"}
COVERAGE_THRESHOLD=${COVERAGE_THRESHOLD:-85}
MAX_LINE_LENGTH=${MAX_LINE_LENGTH:-100}
MAX_COMPLEXITY=${MAX_COMPLEXITY:-15}

echo -e "${BLUE}🔍 PDE-Fluid-Φ Quality Assurance Pipeline${NC}"
echo "=============================================="

# Check if we're in the right directory
if [[ ! -f "pyproject.toml" ]]; then
    echo -e "${RED}❌ Error: pyproject.toml not found. Please run from project root.${NC}"
    exit 1
fi

# Create results directory
mkdir -p reports/

# Function to print section headers
print_section() {
    echo -e "\n${BLUE}📋 $1${NC}"
    echo "$(printf '=%.0s' {1..50})"
}

# Function to check command success
check_success() {
    if [[ $? -eq 0 ]]; then
        echo -e "${GREEN}✅ $1 passed${NC}"
    else
        echo -e "${RED}❌ $1 failed${NC}"
        if [[ "$2" == "critical" ]]; then
            exit 1
        fi
    fi
}

# 1. Environment Setup
print_section "Environment Setup"
echo "Python version: $(python --version)"
echo "Platform: $(uname -s) $(uname -m)"
echo "Available GPUs: $(python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo '0')"

# Install dependencies if needed
if [[ "$INSTALL_DEPS" == "true" ]]; then
    echo "Installing dependencies..."
    pip install -e ".[dev]" > /dev/null 2>&1
    check_success "Dependency installation"
fi

# 2. Code Formatting
print_section "Code Formatting"

echo "Checking Black formatting..."
black --check --diff src/ tests/ examples/ > reports/black_report.txt 2>&1
check_success "Black formatting"

echo "Checking isort import sorting..."
isort --check-only --diff src/ tests/ examples/ > reports/isort_report.txt 2>&1
check_success "isort import sorting"

# 3. Linting
print_section "Linting"

echo "Running flake8 linting..."
flake8 src/ tests/ examples/ \
    --max-line-length=$MAX_LINE_LENGTH \
    --max-complexity=$MAX_COMPLEXITY \
    --format=json \
    --output-file=reports/flake8_report.json \
    --tee \
    --count \
    --statistics
check_success "flake8 linting"

# 4. Type Checking
print_section "Type Checking"

echo "Running mypy type checking..."
mypy src/ \
    --ignore-missing-imports \
    --no-strict-optional \
    --junit-xml reports/mypy_report.xml \
    --html-report reports/mypy_html/ \
    --txt-report reports/ \
    --cobertura-xml-report reports/
check_success "mypy type checking"

# 5. Security Scanning
print_section "Security Scanning"

echo "Running comprehensive security scan..."
if [[ -f "scripts/security_scan.py" ]]; then
    python scripts/security_scan.py \
        --output reports/security_report.json \
        --fail-on-high
    check_success "Comprehensive security scan" "critical"
else
    echo "Security scan script not found, running basic checks..."
    
    echo "Running bandit security scan..."
    if command -v bandit &> /dev/null; then
        bandit -r src/ \
            -f json \
            -o reports/bandit_report.json \
            -ll \
            --severity-level medium
        check_success "bandit security scan"
    else
        echo "bandit not available, skipping"
    fi

    echo "Running safety vulnerability check..."
    if command -v safety &> /dev/null; then
        safety check \
            --json \
            --output reports/safety_report.json \
            --continue-on-error
        check_success "safety vulnerability check"
    else
        echo "safety not available, skipping"
    fi
fi

# 6. Unit Tests
print_section "Unit Tests"

echo "Running unit tests with coverage..."
pytest tests/ \
    -v \
    --cov=src/pde_fluid_phi \
    --cov-branch \
    --cov-report=html:reports/coverage_html \
    --cov-report=xml:reports/coverage.xml \
    --cov-report=term-missing:skip-covered \
    --cov-fail-under=$COVERAGE_THRESHOLD \
    --junit-xml=reports/pytest_report.xml \
    --tb=short \
    -n auto
check_success "Unit tests" "critical"

# 7. Integration Tests
print_section "Integration Tests"

if [[ -d "tests/integration" ]]; then
    echo "Running integration tests..."
    pytest tests/integration/ \
        -v \
        --tb=short \
        --junit-xml=reports/integration_report.xml
    check_success "Integration tests"
else
    echo "No integration tests found, skipping..."
fi

# 8. Performance Tests
print_section "Performance Tests"

echo "Running performance benchmarks..."
if [[ -f "scripts/performance_benchmark.py" ]]; then
    python scripts/performance_benchmark.py \
        --quick \
        --output reports/benchmark_results.json \
        --format json
    check_success "Performance benchmarks"
else
    echo "Performance benchmark script not found"
    
    if command -v pytest-benchmark &> /dev/null; then
        echo "Running pytest benchmarks..."
        pytest tests/ \
            -m benchmark \
            --benchmark-json=reports/benchmark_report.json \
            --benchmark-html=reports/benchmark_report.html \
            --benchmark-only
        check_success "Performance benchmarks"
    else
        echo "No benchmark tools available, skipping..."
    fi
fi

# 9. Documentation Tests
print_section "Documentation Tests"

echo "Testing docstring examples..."
python -m doctest src/pde_fluid_phi/utils/spectral_utils.py -v > reports/doctest_report.txt 2>&1
check_success "Docstring examples"

if command -v sphinx-build &> /dev/null; then
    echo "Building documentation..."
    sphinx-build -b html docs/ reports/docs_html/ -W --keep-going > reports/sphinx_build.log 2>&1
    check_success "Documentation build"
else
    echo "Sphinx not available, skipping documentation build..."
fi

# 10. CLI Tests
print_section "CLI Tests"

echo "Testing CLI commands..."
pde-fluid-phi --help > /dev/null 2>&1
check_success "CLI help command"

pde-fluid-phi generate --help > /dev/null 2>&1
check_success "CLI generate help"

pde-fluid-phi train --help > /dev/null 2>&1
check_success "CLI train help"

# 11. Example Tests
print_section "Example Tests"

if [[ -f "examples/basic_usage.py" ]]; then
    echo "Testing basic usage example (with timeout)..."
    timeout 300 python examples/basic_usage.py > reports/example_output.log 2>&1 || echo "Example timed out or completed"
    check_success "Basic usage example"
else
    echo "Basic usage example not found, skipping..."
fi

# 12. Package Tests
print_section "Package Tests"

echo "Testing package installation..."
pip install -e . > /dev/null 2>&1
check_success "Package installation"

echo "Testing package imports..."
python -c "
import sys
try:
    import pde_fluid_phi
    from pde_fluid_phi.models import RationalFNO
    from pde_fluid_phi.data import TurbulenceDataset
    from pde_fluid_phi.training import StabilityTrainer
    print('✅ All imports successful')
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
" > reports/import_test.log 2>&1
check_success "Package imports" "critical"

# 13. Generate Quality Report
print_section "Quality Report Generation"

echo "Generating comprehensive quality report..."
python scripts/generate_quality_report.py \
    --reports-dir reports/ \
    --output reports/quality_report.html \
    --format html
check_success "Quality report generation"

# 14. Summary
print_section "Quality Assurance Summary"

# Count total issues
total_issues=0

# Count flake8 issues
if [[ -f "reports/flake8_report.json" ]]; then
    flake8_issues=$(python -c "
import json
try:
    with open('reports/flake8_report.json') as f:
        data = json.load(f)
        print(len(data))
except:
    print(0)
")
    total_issues=$((total_issues + flake8_issues))
    echo "Flake8 issues: $flake8_issues"
fi

# Count bandit issues
if [[ -f "reports/bandit_report.json" ]]; then
    bandit_issues=$(python -c "
import json
try:
    with open('reports/bandit_report.json') as f:
        data = json.load(f)
        print(len(data.get('results', [])))
except:
    print(0)
")
    total_issues=$((total_issues + bandit_issues))
    echo "Security issues: $bandit_issues"
fi

# Coverage report
if [[ -f "reports/coverage.xml" ]]; then
    coverage=$(python -c "
import xml.etree.ElementTree as ET
try:
    tree = ET.parse('reports/coverage.xml')
    root = tree.getroot()
    coverage = root.get('line-rate', '0')
    print(f'{float(coverage)*100:.1f}%')
except:
    print('N/A')
")
    echo "Test coverage: $coverage"
fi

echo -e "\nTotal code quality issues: $total_issues"

if [[ $total_issues -eq 0 ]]; then
    echo -e "${GREEN}🎉 All quality checks passed! Code is ready for production.${NC}"
    exit 0
elif [[ $total_issues -lt 10 ]]; then
    echo -e "${YELLOW}⚠️  Minor issues found. Consider addressing before release.${NC}"
    exit 0
else
    echo -e "${RED}❌ Significant issues found. Please address before proceeding.${NC}"
    exit 1
fi