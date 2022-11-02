# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


setup(
    name="3DCORE_h4c",
    packages=find_packages(),
    package_data={'': ['*.json']},
    include_package_data=True,
    version="0.0.0",
    author="Andreas J. Weiss",
    author_email="andreas.weiss@oeaw.ac.at",
    description="3D Coronal Rope Ejection Model",
    keywords=["astrophysics", "solar physics", "space weather"],
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    install_requires=[
        "heliosat",
        "numba",
        "numpy",
        "scipy",
        "spiceypy"
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
