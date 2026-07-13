from setuptools import setup, find_packages
import pathlib

# The directory containing this file
PYNU = pathlib.Path(__file__).parent

# The text of the README file
README = (PYNU / "README.md").read_text()

setup(
    name="pynu",  # Replace with your package name
    version="0.1.0",  # Initial version of your package
    description="A brief description of your package",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/pabloferm/Pynu",  # Replace with your repo
    author="Pablo",
    author_email="pablo.fernandez@dipc.org",
    license="MIT",  # Choose your license
    classifiers=[
        "License :: OSI Approved :: MIT License",  # License type
        # "Programming Language :: Python :: 3",  # Python version compatibility
        # "Programming Language :: Python :: 3.7",
        # "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
        # Additional classifiers can be found at https://pypi.org/classifiers/
    ],
    packages=find_packages(exclude=("tests", "docs")),  # Packages to include
    include_package_data=True,  # Include data files as specified in MANIFEST.in
    package_data={
        # SK binned engine ships its dial-value XMLs as package data so a
        # non-editable wheel carries the dial values (Track S / review N-2:
        # CANONICAL_DIALS was removed at E6, so the XMLs are the sole authority).
        "pynu.binned": [
            "SK2023_Atm_datafit_r2_fude_ccqe_full.xml",
            "SK2023_Atm_datafit_binned_extra_dials.xml",
        ],
    },
    install_requires=[
        # List of dependencies
        "boost_histogram>=1.4.0",
        "h5py>=3.10.0",
        "iminuit>=2.25.2",
        "KDEpy>=1.1.8",
        "matplotlib>=3.5.1",
        "nuflux>=2.0.5",
        "numpy>=1.21.5",
        "pandas>=2.2.0",
        "PyLaTeX>=1.4.2",
        "scipy>=1.8.0",
        # Add more dependencies here
    ],
    extras_require={
        "dev": [
            # "pytest>=6.2.2",
            # "black>=20.8b1",
            # "flake8>=3.8.4",
            # Other development dependencies
        ],
        "docs": [
            "Sphinx>=3.5.1",
            "sphinx-rtd-theme>=0.5.1",
            "sphinxcontrib-napoleon",
            # Other documentation dependencies
        ],
    },
    entry_points={
        "console_scripts": [
            "your-command=your_package.module:function",  # Command-line executable
        ],
    },
    python_requires=">=3.7",  # Minimum Python version requirement
    project_urls={  # Additional links for the project
        # "Bug Tracker": "https://github.com/yourusername/your-repo-name/issues",
        # "Documentation": "https://pablofer.github.io/your-repo-name",
        "Source Code": "https://github.com/pabloferm/Pynu",
    },
)
