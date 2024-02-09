from setuptools import find_packages, setup
"""
This Pip skeleton was created by following Michael Kim's Pip 
package tutorial:
    - https://github.com/MichaelKim0407/tutorial-pip-package
"""

setup(
    name="MTH205",
    description="A package extension for sage.",
    version="0.0.2.dev1",
    url="https://github.com/davidaustinm/MTH205-W20",
    author="GVSU MTH205",
    packages=find_packages(
        include='mth205'
    ),
    install_requires=[
        "numpy",
        "matplotlib",
        "pillow",
        "scipy",
        "pandas"
    ]
)