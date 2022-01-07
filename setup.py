from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="jaxlie",
    version="1.2.10",
    description="Matrix Lie groups in Jax",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="http://github.com/brentyi/jaxlie",
    author="brentyi",
    author_email="brentyi@berkeley.edu",
    license="MIT",
    packages=find_packages(),
    package_data={"jaxlie": ["py.typed"]},
    python_requires=">=3.7",
    install_requires=[
        "jax>=0.2.14",  # jax==0.2.14 introduces `ndarray.at[].multiply()`.
        "jaxlib>=0.1.67",
        "jax_dataclasses>=1.2.0",  # Required for shape/data-type annotations.
        "numpy",
        "overrides!=4",
    ],
    extras_require={
        "testing": [
            "flax",
            "hypothesis[numpy]",
            "pytest",
            "pytest-xdist[psutil]",
            "pytest-cov",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
