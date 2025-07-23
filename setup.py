from setuptools import setup, find_packages

setup(
    name="sampler",
    version="0.1",
    description="A package for raster and vector clipping.",
    author="Salar Ghaffarian",
    author_email="salar.ghaffarian1@gmail.com",
    packages=find_packages(),
    install_requires=[ "gdal","numpy"],
    python_requires=">=3.7", 
)