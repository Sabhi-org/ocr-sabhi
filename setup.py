import setuptools

requirements = []
with open('requirements.txt', 'r') as fh:
    for line in fh:
        requirements.append(line.strip())

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ocr-sabhi",
    version="0.0.4",
    author="Hamza",
    author_email="info@sabhi.org",
    description="A simple document detector in python.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sabhi-org/ocr-sabhi",
    packages=setuptools.find_packages(),

    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
