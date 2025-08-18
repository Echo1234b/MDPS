#!/usr/bin/env python3
"""
MDPS - Market Data Processing System
Setup and installation configuration
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = []
with open(this_directory / "requirements.txt", "r", encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#') and not line.startswith('Note:'):
            requirements.append(line)

setup(
    name="mdps",
    version="1.0.0",
    author="MDPS Development Team",
    author_email="dev@mdps.com",
    description="Advanced Market Data Processing System with ML predictions and trading strategies",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mdps/mdps",
    project_urls={
        "Bug Tracker": "https://github.com/mdps/mdps/issues",
        "Documentation": "https://mdps.readthedocs.io/",
        "Source Code": "https://github.com/mdps/mdps",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    packages=find_packages(include=["mdps", "mdps.*"]),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.19.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "isort>=5.10.0",
        ],
        "gpu": [
            "tensorflow-gpu>=2.8.0",
            "torch>=1.12.0",
            "cuda-python>=11.7.0",
        ],
        "full": [
            "prophet>=1.1.0",
            "pmdarima>=1.8.0",
            "pycaret>=3.0.0",
            "web3>=5.31.0",
            "newsapi-python>=0.2.6",
        ],
    },
    entry_points={
        "console_scripts": [
            "mdps=mdps.run_mdps:main",
            "mdps-validate=mdps.run_mdps:run_validation_mode",
            "mdps-test=mdps.run_mdps:run_test_mode",
        ],
    },
    include_package_data=True,
    package_data={
        "mdps": ["config/*.yaml", "config/*.yml", "*.md", "LICENSE"],
    },
    zip_safe=False,
    keywords="finance, trading, machine-learning, data-analysis, market-data, technical-analysis",
)