from setuptools import setup, find_packages # type: ignore 

setup(
    name="DeepTranslate",
    version="0.1.0",
    author="Kandarpa Sarkar",
    author_email="kandarpaexe@gmail.com",
    description="Machine translation model based on 2017 paper by google 'Attention is all you need'",
    long_description=open("README.md").read(),
    url="https://github.com/kandarpa02/DeepTranslate.git",
    packages=find_packages(),
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)