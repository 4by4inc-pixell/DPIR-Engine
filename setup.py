"""
setup script for pip install
"""

from setuptools import setup, find_packages


REQUIREMENTS = [
    "onnx",
    "onnxruntime-gpu",
    "ortei",
    "torch",
]

INCLUDE_MODULE = ["dpir"]


setup(
    name="dpir-engine",
    version="0.1.0",
    packages=find_packages(include=INCLUDE_MODULE),
    install_requires=REQUIREMENTS,
)
