from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="labelme-sam2-propagate",
    version="0.1.0",
    author="hatmore",
    description="SAM2 video propagation tool for LabelMe - accelerate annotation with cross-frame tracking",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hatmore/labelme-sam2-propagate",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "opencv-python>=4.8.0",
        "Pillow>=10.0.0",
        "sam2>=1.0.0",
        "torch>=2.0.0",
    ],
    entry_points={
        "console_scripts": [
            "labelme-sam2-propagate=labelme_sam2_propagate.cli:main",
        ],
    },
)
