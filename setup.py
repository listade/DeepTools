from setuptools import setup, find_packages

setup(
    name="detect",
    version="0.0.1",
    author="Sholukh Egor",
    author_email="egor.bulochka@gmail.com",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "matplotlib==3.5.2",
        "numpy==1.21.6",
        "opencv-python==4.6.0.66",
        "Pillow==9.2.0",
        "pyaml==21.10.1",
        "scipy==1.7.3",
        "tqdm==4.64.0",
        "torch",
        "torchaudio",
        "torchvision",
    ],
)
