"""Setup script for MSA-T OSV package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="msa_t_osv",
    version="0.1.0",
    author="MSA-T OSV Contributors",
    author_email="msa-t-osv@example.com",
    description="Multi-Scale Attention and Transformer-based Offline Signature Verification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/msa_t_osv",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/msa_t_osv/issues",
        "Documentation": "https://github.com/yourusername/msa_t_osv/wiki",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "flake8>=6.0.0",
            "black>=23.3.0",
            "isort>=5.12.0",
        ],
        "visualization": [
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "msa-t-osv-train=msa_t_osv.train:main",
            "msa-t-osv-evaluate=msa_t_osv.evaluate:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
) 