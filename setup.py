from setuptools import setup, find_packages

setup(
    name="regression_analysis",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.1.0",
        "numpy>=1.24.3",
        "scikit-learn>=1.3.0",
        "openpyxl>=3.1.2",
        "matplotlib>=3.7.1",
        "seaborn>=0.12.2",
    ],
)
