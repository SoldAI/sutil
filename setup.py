import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="soldai-utils",
    version="0.0.1",
    author="Soldai Research",
    author_email="mcampos@soldai.com",
    description="Soldai utilities for machine learning and text processing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SoldAI/sutil",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
        python_requires='>=3.6',
)
