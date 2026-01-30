# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 11:44:51 2026

@author: m.roska
"""

from setuptools import setup, find_packages

setup(
    name="VoltageScanning",
    version="0.1.0",
    description="Voltage Scanning Analysis for CIMS quantification",
    author="Milan Roska",
    packages=find_packages(),  # Finds vsfit/, and anything inside
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "matplotlib"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
