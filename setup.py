"""Setup script for backward compatibility."""

from setuptools import setup, find_packages

setup(
    name="labelme-sam2-propagate",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "Pillow>=10.0.0",
        "sam2>=1.0.0",
    ],
    entry_points={
        "console_scripts": [
            "labelme-sam2-propagate=labelme_sam2_propagate.cli:main",
        ],
    },
    python_requires=">=3.9",
)
