from setuptools import setup, find_packages

setup(
    name="deep_osr",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=1.10.0",
        "torchvision>=0.11.0",
        "pytorch-lightning>=1.5.0",
        "hydra-core>=1.1.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "timm>=0.5.0",
        "PyYAML>=5.4.0",
        "omegaconf>=2.1.0",
    ],
    entry_points={
        "console_scripts": [
            "osr_train=train:main", # Assuming train.py has main decorated with @hydra.main
            "osr_eval=eval:main",   # Assuming eval.py has main decorated with @hydra.main
        ]
    },
)