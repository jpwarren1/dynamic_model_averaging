from setuptools import setup, find_packages

setup(
    name='dma',
    version='0.1dev',
    packages= find_packages(where='src'),
    package_dir={"":"src"}
)