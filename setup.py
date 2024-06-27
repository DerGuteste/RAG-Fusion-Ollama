from setuptools import setup, find_packages

setup(
    name='search-fusion',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'ollama>=0.2.1',
        'requests>=2.32.3',
        'tqdm>=4.66.4',
        'faiss-cpu>=1.8.0',
        'numpy>=1.26.4'
    ],
)