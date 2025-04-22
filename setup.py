from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="BERTune",
    version="0.1.0",
    author="Abhay Shastry",
    author_email="abhay.s.shastry@gmail.com",
    description="A CLI tool for fine-tuning BERT models on custom datasets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AbhayShas3/BERTune",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "bertune=bertune.cli:main",
        ],
    },
)