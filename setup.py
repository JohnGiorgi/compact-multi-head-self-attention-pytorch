import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="lama",
    version="0.1.0",
    author="John Giorgi",
    author_email="johnmgiorgi@gmail.com",
    description=("A PyTorch implementation of the Compact Multi-Head Self-Attention Mechanism from"
                 " the paper: 'Low Rank Factorization for Compact Multi-Head Self-Attention'"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JohnGiorgi/compact-multi-head-self-attention-pytorch",
    packages=setuptools.find_packages(),
    keywords=["natural language processing", "pytorch", "self-attention", "multi-head attention"],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Typing :: Typed"
    ],
    python_requires='>=3.7',
    install_requires=[
        'torch>=1.2.0',
    ],
    extras_require={
        'dev': [
            'tox',
            'pytest',
            'pytest-cov'
        ]
    }
)
