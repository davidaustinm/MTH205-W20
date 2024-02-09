from setuptools import find_packages, setup

setup(
    name="mth205",
    description="A package extension for sage.",
    version="0.0.3",
    url="https://github.com/davidaustinm/MTH205-W20",
    author="GVSU MTH205",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "pillow",
        "scipy",
        "pandas"
    ]
)