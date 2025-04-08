from setuptools import setup, find_packages

setup(
    name="rss_polymlp",
    version="0.0.1",
    license="MIT",
    description="A framework for random structure search using polynomial MLPs",
    long_description="",
    long_description_content_type="",
    author="Hayato Wakai",
    author_email="wakai@cms.mtl.kyoto-u.ac.jp",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={"rss_polymlp": ["py.typed"]},
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "scipy",
        "setuptools",
        "pyyaml",
        "pybind11",
        "joblib",
        "pypolymlp",
        "spglib",
        "symfc",
    ],
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
