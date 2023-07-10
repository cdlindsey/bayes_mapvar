'''Setup file for the revpy module'''

from setuptools import setup, find_packages


def _load_requirements():
    with open('requirements.txt', 'r') as rfile:
        return rfile.read().strip().split('\n')


setup(
    name='bayes_mapvar',
    description='Bayesian Maximum a Posteriori/Variance estimation',
    version='0.0.1',
    author='Charles Lindsey',
    author_email='lindseycster@gmail.com',
    long_description_content_type="text/markdown",
    long_description='README.md',
    packages=find_packages(where='src', exclude=['tests', 'tests.*']),
    package_dir={'': 'src'},
    install_requires=_load_requirements(),
)
