from setuptools import setup, find_packages

setup(
    name="detect",
    version="0.0.1",
    author="Sholukh Egor",
    author_email="egor.bulochka@gmail.com",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=open("requirements.txt").read().split("\n"),
)
