from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="jaxlie",
    version="1.2.1",
    description="Matrix Lie groups in Jax",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="http://github.com/brentyi/jaxlie",
    author="brentyi",
    author_email="brentyi@berkeley.edu",
    license="MIT",
    packages=find_packages(),
    package_data={"jaxlie": ["py.typed"]},
    python_requires=">=3.6",
    install_requires=[
        "flax",
        "jax",
        "jaxlib",
        "numpy",
        # `overrides` should not be updated until the following issues are resolved:
        # > https://github.com/mkorpela/overrides/issues/65
        # > https://github.com/mkorpela/overrides/issues/63
        # > https://github.com/mkorpela/overrides/issues/61
        "overrides<4",
        "dataclasses; python_version < '3.7.0'",
    ],
    extras_require={
        "testing": [
            "pytest",
            "pytest-cov",
            "hypothesis",
            "hypothesis[numpy]",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
