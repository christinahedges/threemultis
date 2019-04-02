#!/usr/bin/env python
import os
import sys
from setuptools import setup

# Prepare and send a new release to PyPI
if "release" in sys.argv[-1]:
    os.system("python setup.py sdist")
    os.system("twine upload dist/*")
    os.system("rm -rf dist/threemultis*")
    sys.exit()

# Load the __version__ variable without importing the package already
exec(open('threemultis/version.py').read())

# DEPENDENCIES
# 1. What are the required dependencies?
with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

setup(name='threemultis',
      version=__version__,
      description="An unfriendly package for chromatic PSFs in TESS",
      long_description=open('README.md').read(),
      author='TESS Party',
      author_email='christina.l.hedges@nasa.gov',
      license='MIT',
      package_dir={
            'threemultis': 'threemultis'},
      packages=['threemultis'],
      install_requires=install_requires,
      setup_requires=['pytest-runner'],
      include_package_data=True,
      classifiers=[
          "Development Status :: 0 - Rubbish",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
          "Programming Language :: Python",
          "Intended Audience :: Science/Research",
          "Topic :: Scientific/Engineering :: Astronomy",
          ],
      )
