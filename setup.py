import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="rnt",
    version="0.1.1",
    author="Jacob Rohde",
    author_email="jarohde1@gmail.com",
    description="A simple tool for generating and analyzing Reddit networks.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jarohde/rnt",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    install_requires=['networkx', 'pandas', 'pmaw']
)
