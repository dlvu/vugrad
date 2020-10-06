from setuptools import setup

import sys
if sys.version_info < (3,8):
    sys.exit('A python version of 3.8 or higher is required. Consider creating a ptyhon virtual environment.')

setup(
    name="vugrad",
    version="0.1",
    description="A minimal autodiff library",
    url="http://dlvu.github.io",
    author="Peter Bloem (Vrije Universiteit)",
    author_email="vugrad@peterbloem.nl",
    packages=["vugrad"],
    install_requires=[
        "numpy",
    ],
    zip_safe=False,
    entry_points={"console_scripts": ["kge = kge.cli:main",],},
)
