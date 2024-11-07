from setuptools import setup, find_packages

setup(
    name="clipper",
    version="0.1.0",
    description="A package for raster and vector clipping.",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author="Salar Ghaffarian",
    author_email="salar@myheat.ca",
    url="https://github.com/salarghaffarian/clipper", 
    packages=find_packages(where='clipper'),
    install_requires=[ "gdal",
                       "numpy"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Windows, Linux, MacOS",
    ],
    python_requires=">=3.6", 
)