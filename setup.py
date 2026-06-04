"""
Setup configuration for PayGuard - AI-Powered Payment Fraud Detection.

This file defines the package metadata and installation configuration
for the PayGuard data science project.
"""

from pathlib import Path
from typing import List

from setuptools import find_packages, setup

PROJECT_ROOT = Path(__file__).resolve().parent


def read_requirements(file_path: str = "requirements.txt") -> List[str]:
    """
    Read project dependencies from a requirements file.

    Args:
        file_path: Path to the requirements file.

    Returns:
        A list of dependency names.

    Raises:
        FileNotFoundError: If the requirements file does not exist.
    """
    requirements_path = PROJECT_ROOT / file_path

    if not requirements_path.exists():
        raise FileNotFoundError(f"Requirements file not found: {requirements_path}")

    return [
        line.strip()
        for line in requirements_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]


setup(
    name="payguard",
    version="0.1.0",
    author="Ridhan Parvendhan",
    author_email="ridhanparvendhan@gmail.com",
    description="AI-powered payment fraud detection using machine learning.",
    long_description=(
        "PayGuard is a professional Python data science project designed to "
        "detect fraudulent payment transactions using machine learning, "
        "class imbalance handling, explainability, and an interactive dashboard."
    ),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=read_requirements(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial",
        "Operating System :: OS Independent",
    ],
    keywords=[
        "fraud-detection",
        "machine-learning",
        "data-science",
        "payment-fraud",
        "xgboost",
        "shap",
        "streamlit",
    ],
    include_package_data=True,
)
