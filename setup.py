# References: https://the-hitchhikers-guide-to-packaging.readthedocs.io/en/latest/quickstart.html
# Single sourcing version follows: https://github.com/pypa/pip/blob/master/setup.py#L12

from setuptools import setup, find_packages
import os
import codecs


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    # intentionally *not* adding an encoding option to open, See:
    #   https://github.com/pypa/virtualenv/issues/201#issuecomment-3145690
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


with open('requirements.txt', 'r') as reqfile:
    requirements = [package_name.replace('\n', '') for package_name in reqfile]


setup(
    name='prophet_gridsearch',
    version=get_version("prophet_gridsearch/__init__.py"),
    packages=find_packages(),
    author='Raúl Gadea Pérez',
    author_email='raul.gadea@sdggroup.com',
    long_description=open('README.md').read(),
    install_requires=requirements,
    python_requires='>=3.6',
    include_package_data=True  # we add this option to be able to include non-python files in the package distribution
)
