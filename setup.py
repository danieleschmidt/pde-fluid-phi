"""
PDE-Fluid-Φ: N-Dimensional Neural Operators for High-Reynolds Number Turbulent Flows
"""

from setuptools import setup, find_packages
import os

# Read long description from README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements from requirements.txt
def read_requirements(filename):
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return []

setup(
    name="pde-fluid-phi",
    version="0.1.0",
    author="Terragon Labs Research Team",
    author_email="research@terragonlabs.com",
    description="N-Dimensional Neural Operators for High-Reynolds Number Turbulent Flows",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/terragonlabs/pde-fluid-phi",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "cuda": [
            "cupy-cuda11x>=10.0.0",
            "triton>=2.0.0",
        ],
        "viz": [
            "matplotlib>=3.5.0",
            "plotly>=5.0.0",
            "mayavi>=4.8.0",
            "pyvista>=0.37.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
            "pre-commit>=2.20.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.18.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "pde-fluid-phi=pde_fluid_phi.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)