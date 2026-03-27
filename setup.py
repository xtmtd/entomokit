from setuptools import setup, find_packages

setup(
    name="entomokit",
    version="0.1.6",
    description="A Python toolkit for building insect image datasets with segmentation, frame extraction, cleaning, dataset splitting, and image synthesis capabilities",
    author="Feng ZHANG",
    author_email="xtmtd.zf@gmail.com",
    url="https://github.com/xtmtd/entomokit",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "Pillow>=10.0.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "segmentation": [
            "torch>=2.0.0,<2.4.0",
            "torchvision>=0.15.0,<0.19.0",
            "opencv-python>=4.8.0",
            "scikit-image>=0.21.0",
        ],
        "cleaning": [
            "imagehash",
        ],
        "video": [
            "opencv-python>=4.8.0",
        ],
        "data": [
            "pandas",
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
        ],
        "augment": [
            "albumentations>=1.4.0",
        ],
        "classify": [
            "autogluon.multimodal>=1.4.0",
            "timm>=0.9.0",
            "umap-learn",
            "pytorch-grad-cam",
            "onnxruntime",
            "scikit-learn",
        ],
    },
    python_requires=">=3.8",
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
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
)
