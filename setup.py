from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="src",
    version="0.0.1",
    author="Manav664",
    description="A small package for dvc DL pipeline demo",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Manav664/DVC-DL-Project.git",
    author_email="sunny.c17hawke@gmail.com",
    # package_dir={"": "src"},
    # packages=find_packages(where="src"),
    packages=["src"],
    license="GNU",
    python_requires=">=3.6",
    install_requires=[
        'dvc',
        'pandas',
        'tensorflow',
        'matplotlib',
        'tqdm',
        'PyYAML',
        'boto3'
    ]
)