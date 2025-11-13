from setuptools import setup
from pathlib import Path
import sys


package_dir = Path(__file__).parent / "mlx_embeddings_lora"
with open("requirements.txt") as fid:
    requirements = [l.strip() for l in fid.readlines()]

sys.path.append(str(package_dir))
from version import __version__

setup(
    name="mlx-embeddings-lora",
    version=__version__,
    description="Train Embedding Models on Apple silicon with MLX and the Hugging Face Hub",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    readme="README.md",
    author_email="goekdenizguelmez@gmail.com",
    author="Gökdeniz Gülmez",
    url="https://github.com/Goekdeniz-Guelmez/mlx-embeddings-lora",
    license="MIT",
    install_requires=requirements,
    packages=["mlx_embeddings_lora", "mlx_embeddings_lora.trainer"],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "mlx_embeddings_lora.train = mlx_embeddings_lora.train:main",
        ]
    },
)
