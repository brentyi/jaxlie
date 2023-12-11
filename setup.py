from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="jaxlie",
    version="1.3.4",
    description="Matrix Lie groups in JAX",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="http://github.com/brentyi/jaxlie",
    author="brentyi",
    author_email="brentyi@berkeley.edu",
    license="MIT",
    packages=find_packages(),
    package_data={"jaxlie": ["py.typed"]},
    python_requires=">=3.8",
    install_requires=[
        "jax>=0.3.18",  # For jax.Array.
        "jax_dataclasses>=1.4.4",
        "numpy",
        "tyro",  # Only used in examples.
    ],
    extras_require={
        "testing": [
            "mypy",
            # https://github.com/google/jax/issues/12536
            "jax!=0.3.19",
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
