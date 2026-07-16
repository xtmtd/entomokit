from setuptools import setup, find_packages

setup(
    name="entomokit",
    version="0.5.0",
    description="A Python toolkit for building insect image datasets with segmentation, frame extraction, cleaning, dataset splitting, and image synthesis capabilities",
    author="Feng ZHANG",
    author_email="xtmtd.zf@gmail.com",
    url="https://github.com/xtmtd/entomokit",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "Pillow>=10.0.0",
        "tqdm>=4.65.0",
        "pandas",
    ],
    extras_require={
        "segmentation": [
            "torch>=2.0.0",
            "torchvision>=0.15.0",
            "opencv-python>=4.8.0",
            "scikit-image>=0.21.0",
            "scipy>=1.10.0",
            "iopath",
            "huggingface-hub",
            "einops",
            "pycocotools",
            "ftfy",
        ],
        "synthesis": [
            "supervision>=0.22.0",
            "scipy>=1.10.0",
        ],
        "measurement": [
            "scipy>=1.10.0",
        ],
        "cleaning": [
            "imagehash",
        ],
        "video": [
            "opencv-python>=4.8.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
        ],
        "augment": [
            "albumentations>=1.4.0",
        ],
        "classify": [
            "autogluon.multimodal>=1.5.0",
            "timm>=0.9.0",
            "umap-learn",
            "matplotlib",
            "seaborn",
            "grad-cam",
            "onnxruntime",
            "onnx",
            "scikit-learn",
            "setuptools<70",   # ponytail: AutoGluon needs pkg_resources removed in >=70
        ],
    },
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "entomokit=entomokit.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
)
